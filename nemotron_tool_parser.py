# SPDX-License-Identifier: Apache-2.0
# Custom vLLM tool parser for NVIDIA Nemotron models.
#
# Nemotron emits tool calls in this XML format:
#
#   <tool_call>
#   <function=function_name>
#   <parameter=param1>
#   value1
#   </parameter>
#   <parameter=param2>
#   value2
#   </parameter>
#   </function>
#   </tool_call>
#
# No built-in vLLM parser handles this format. The "hermes" parser expects
# JSON inside <tool_call> tags; "pythonic" expects Python function-call syntax.
# This parser bridges the gap so that vLLM can extract structured tool_call
# objects from Nemotron's output, which are then translated into Anthropic
# tool_use blocks by vLLM's /v1/messages compatibility layer.

import json
from collections.abc import Sequence

import regex as re
from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
)

logger = init_logger(__name__)

# Regex to capture individual tool calls (greedy between tags)
TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL
)

# Regex to extract function name from <function=name>
FUNCTION_NAME_RE = re.compile(r"<function=([^>]+)>")

# Regex to extract parameters: <parameter=key>value</parameter>
PARAMETER_RE = re.compile(
    r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL
)


def parse_nemotron_tool_call(block: str) -> dict:
    """Parse a single Nemotron tool call block into a dict with
    'name' and 'arguments' keys."""
    name_match = FUNCTION_NAME_RE.search(block)
    if not name_match:
        raise ValueError(f"No function name found in tool call block: {block}")

    name = name_match.group(1).strip()
    arguments = {}

    for param_match in PARAMETER_RE.finditer(block):
        param_name = param_match.group(1).strip()
        param_value = param_match.group(2).strip()
        # Try to parse as JSON value (for numbers, bools, objects, arrays)
        try:
            param_value = json.loads(param_value)
        except (json.JSONDecodeError, ValueError):
            pass  # keep as string
        arguments[param_name] = param_value

    return {"name": name, "arguments": arguments}


class NemotronToolParser(ToolParser):

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)

        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = []

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:

        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            matches = TOOL_CALL_BLOCK_RE.findall(model_output)
            raw_blocks = [m[0] if m[0] else m[1] for m in matches]

            tool_calls = []
            for block in raw_blocks:
                parsed = parse_nemotron_tool_call(block)
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=parsed["name"],
                            arguments=json.dumps(
                                parsed["arguments"], ensure_ascii=False
                            ),
                        ),
                    )
                )

            # Content is everything before the first <tool_call>
            content = model_output[
                : model_output.find(self.tool_call_start_token)
            ].strip()

            # Strip thinking tags if present
            content = re.sub(
                r"</?think>", "", content
            ).strip()

            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content else None,
            )

        except Exception:
            logger.exception(
                "Error extracting Nemotron tool calls from response."
            )
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:

        # If no tool call tag seen yet, stream as text content
        if self.tool_call_start_token not in current_text:
            return DeltaMessage(content=delta_text)

        # If we just hit the start token, begin tracking a new tool call
        if (
            self.tool_call_start_token in delta_text
            or (
                self.tool_call_start_token in current_text
                and self.tool_call_start_token not in previous_text
            )
        ):
            self.current_tool_id += 1
            self.current_tool_name_sent = False
            self.streamed_args_for_tool.append("")
            self.prev_tool_call_arr.append({})
            # Send any text content before the tool call
            before = current_text[
                : current_text.find(self.tool_call_start_token)
            ].strip()
            before = re.sub(r"</?think>", "", before).strip()
            if before and self.current_tool_id == 0:
                return DeltaMessage(content=before)
            return None

        # If the end token just appeared, finalize the tool call
        if self.tool_call_end_token in delta_text:
            # Extract the full tool call block
            tool_start = current_text.rfind(self.tool_call_start_token)
            tool_end = current_text.rfind(self.tool_call_end_token)
            if tool_start >= 0 and tool_end > tool_start:
                block = current_text[
                    tool_start + len(self.tool_call_start_token) : tool_end
                ]
                try:
                    parsed = parse_nemotron_tool_call(block)
                    args_json = json.dumps(
                        parsed["arguments"], ensure_ascii=False
                    )

                    if not self.current_tool_name_sent:
                        self.current_tool_name_sent = True
                        return DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    type="function",
                                    id=make_tool_call_id(),
                                    function=DeltaFunctionCall(
                                        name=parsed["name"],
                                        arguments=args_json,
                                    ).model_dump(exclude_none=True),
                                )
                            ]
                        )
                    else:
                        return DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    function=DeltaFunctionCall(
                                        arguments=args_json,
                                    ).model_dump(exclude_none=True),
                                )
                            ]
                        )
                except Exception:
                    logger.exception(
                        "Error parsing streamed Nemotron tool call."
                    )
                    return None
            return None

        # We're inside a tool call block but haven't hit the end yet.
        # Try to extract the function name if we haven't sent it.
        if not self.current_tool_name_sent:
            tool_start = current_text.rfind(self.tool_call_start_token)
            partial_block = current_text[
                tool_start + len(self.tool_call_start_token) :
            ]
            name_match = FUNCTION_NAME_RE.search(partial_block)
            if name_match:
                self.current_tool_name_sent = True
                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            type="function",
                            id=make_tool_call_id(),
                            function=DeltaFunctionCall(
                                name=name_match.group(1).strip(),
                            ).model_dump(exclude_none=True),
                        )
                    ]
                )

        # Otherwise, we're accumulating parameters - don't stream partial args
        return None

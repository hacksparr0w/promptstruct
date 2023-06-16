import json

from dataclasses import asdict
from typing import Iterable, Optional

import openai

from ._llm import (
    AssistantMessage,
    Function,
    FunctionCall,
    LanguageModel,
    Message,
    MessageRole
)


__all__ = [
    "OpenAiLanguageModel"
]


def _serialize_function_call(function_call: FunctionCall) -> dict:
    return {
        "name": function_call.name,
        "arguments": json.dumps(function_call.arguments)
    }


def _serialize_message(message: Message) -> dict:
    role = message.role.name.lower()
    data = {"role": role, "content": message.content}

    if (
        message.role is MessageRole.ASSISTANT and
        message.function_call is not None
    ):
        data["function_call"] = _serialize_function_call(
            message.function_call
        )

    return data


def _deserialize_function_call(data: dict) -> FunctionCall:
    name = data["name"]
    arguments = json.loads(data["arguments"])

    return FunctionCall(name, arguments)


def _deserialize_message(data: dict) -> Message:
    role = MessageRole.of(data["role"].upper())
    content = data["content"]

    match role:
        case MessageRole.ASSISTANT:
            function_call_data = data.get("function_call")
            function_call = None
            
            if function_call_data is not None:
                function_call = _deserialize_function_call(
                    function_call_data
                )

            return Message.assistant(content, function_call)
        case MessageRole.SYSTEM:
            return Message.system(content)
        case MessageRole.USER:
            return Message.user(content)


class OpenAiLanguageModel(LanguageModel):
    def __init__(self, api_key: str, model_name: str) -> None:
        self._api_key = api_key
        self._model_name = model_name
    
    def supports_functions(self) -> bool:
        return self._model_name in ("gpt-4-0613", "gpt-3.5-turbo-0613")

    def prompt(
        self,
        messages: Iterable[Message],
        functions: Optional[Iterable[Function]] = None
    ) -> AssistantMessage:
        kwargs = {
            "api_key": self._api_key,
            "model": self._model_name,
            "messages":  [_serialize_message(message) for message in messages]
        }

        if functions is not None:
            kwargs["functions"] = [asdict(function) for function in functions]

        response = openai.ChatCompletion.create(**kwargs)

        message_data = response["choices"][0]["message"]
        message = _deserialize_message(message_data)

        return message

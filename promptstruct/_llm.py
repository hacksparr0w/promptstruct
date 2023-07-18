from dataclasses import dataclass
from enum import Enum, auto

from typing import Iterable, Optional, Union


@dataclass
class Function:
    name: str
    description: str
    parameters: dict


@dataclass
class FunctionCall:
    name: str
    arguments: dict


class MessageRole(Enum):
    ASSISTANT = auto()
    SYSTEM = auto()
    USER = auto()

    @classmethod
    def of(cls, name: str) -> "MessageRole":
        for role in cls:
            if role.name == name:
                return role

        raise ValueError(f"Invalid message role: '{name}'")


@dataclass
class AssistantMessage:
    content: Optional[str]
    function_call: Optional[FunctionCall]
    role: MessageRole = MessageRole.ASSISTANT


@dataclass
class SystemMessage:
    content: str
    role: MessageRole = MessageRole.SYSTEM


@dataclass
class UserMessage:
    content: str
    role: MessageRole = MessageRole.USER


Message = Union[AssistantMessage, SystemMessage, UserMessage]


def _build_assistant_message(
    content: Optional[str],
    function_call: Optional[FunctionCall] = None,
) -> AssistantMessage:
    return AssistantMessage(content, function_call)


def _build_system_message(content: str) -> SystemMessage:
    return SystemMessage(content)


def _build_user_message(content: str) -> UserMessage:
    return UserMessage(content)


Message.__dict__["assistant"] = _build_assistant_message
Message.__dict__["system"] = _build_system_message
Message.__dict__["user"] = _build_user_message


class LanguageModel:
    def supports_functions(self) -> bool:
        raise NotImplementedError

    async def prompt(
        messages: Iterable[Message],
        functions: Optional[Iterable[Function]] = None
    ) -> AssistantMessage:
        raise NotImplementedError

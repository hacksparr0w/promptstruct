from typing import Iterable, Optional

from ._llm import AssistantMessage, LanguageModel, Message, Function


class LanguageModelWithMemory(LanguageModel):
    def __init__(self, llm: LanguageModel) -> None:
        self._llm = llm
        self._memory = []

    def supports_functions(self) -> bool:
        return self._llm.supports_functions()

    async def prompt(
        self,
        messages: Iterable[Message],
        functions: Optional[Iterable[Function]] = None
    ) -> AssistantMessage:
        memory = [*self._memory, *messages]
        response = await self._llm.prompt(memory, functions)
        memory = [*memory, response]
        self._memory = memory

        return response


def with_memory(llm: LanguageModel) -> LanguageModelWithMemory:
    return LanguageModelWithMemory(llm)

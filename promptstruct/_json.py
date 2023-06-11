from typing import Any

from ._llm import LanguageModel, PromptStrategy


def _prompt_json_primitive(
        llm: LanguageModel,
        strategy: PromptStrategy,
        type: str,
        prompt: str
) -> Any:
    return strategy.prompt(llm, prompt, type)


def _prompt_json_object(
        llm: LanguageModel,
        strategy: PromptStrategy,
        properties: dict
) -> Any:
    result = {}

    for name, schema in properties.items():
        result[name] = _prompt_json_entity(llm, strategy, schema)

    return result


def _prompt_json_entity(
        llm: LanguageModel,
        strategy: PromptStrategy,
        schema: dict
) -> Any:
    type = schema["type"]

    match type:
        case "object":
            properties = schema["properties"]

            return _prompt_json_object(llm, strategy, properties)
        case "string" | "number" | "integer" | "boolean":
            prompt = schema["prompt"]

            return _prompt_json_primitive(llm, strategy, type, prompt)
        case _:
            raise ValueError(f"Unsupported type: {type}")


def json_prompt(
        llm: LanguageModel,
        strategy: PromptStrategy,
        schema: dict
) -> Any:
    return _prompt_json_entity(llm, strategy, schema)

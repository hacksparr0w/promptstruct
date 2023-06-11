from typing import Any

from ._llm import LanguageModel, PromptStrategy


class JsonPromptError(Exception):
    pass


def _prompt_json_primitive(
        llm: LanguageModel,
        strategy: PromptStrategy,
        type: str,
        prompt: str
) -> Any:
    result = strategy.prompt(llm, prompt, type).content

    match type:
        case "string":
            return result
        case "number":
            return float(result)
        case "integer":
            return int(result)
        case "boolean":
            return bool(result)
        case _:
            raise ValueError


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

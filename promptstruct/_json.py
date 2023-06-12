from typing import Any, Iterable

from ._llm import LanguageModel, Message


class PromptBuilder:
    def build_prompt(schema: dict) -> str:
        raise NotImplementedError


class ResponseParser:
    def parse_response(response: Message, schema: dict) -> Any:
        raise NotImplementedError


class DefaultPromptBuilder(PromptBuilder):
    def build_prompt(schema: dict) -> Iterable[Message]:
        return [Message.user(schema["prompt"])]


class DefaultResponseParser(ResponseParser):
    def parse_response(response: Message, schema: dict) -> Any:
        content = response.content
        type = schema["type"]

        match type:
            case "string":
                return content
            case "number":
                return float(content)
            case "integer":
                return int(content)
            case "boolean":
                return bool(content)
            case _:
                raise NotImplementedError


def _prompt_json_primitive(
        llm: LanguageModel,
        prompt_builder: PromptBuilder,
        response_parser: ResponseParser,
        schema: dict
) -> Any:
    prompt = prompt_builder.build_prompt(schema)
    response = llm.prompt(prompt)
    result = response_parser.parse_response(response, schema)

    return result


def _prompt_json_object(
        llm: LanguageModel,
        prompt_builder: PromptBuilder,
        response_parser: ResponseParser,
        parent_schema: dict
) -> Any:
    properties = parent_schema["properties"]
    result = {}

    for name, child_schema in properties.items():
        result[name] = _prompt_json_entity(
            llm,
            prompt_builder,
            response_parser,
            child_schema
        )

    return result


def _prompt_json_entity(
        llm: LanguageModel,
        prompt_builder: PromptBuilder,
        response_parser: ResponseParser,
        schema: dict
) -> Any:
    type = schema["type"]

    match type:
        case "object":
            return _prompt_json_object(
                llm,
                prompt_builder,
                response_parser,
                schema
            )
        case "string" | "number" | "integer" | "boolean":
            return _prompt_json_primitive(
                llm,
                prompt_builder,
                response_parser,
                schema
            )
        case _:
            raise ValueError(f"Unknown type: '{type}'")


def json_prompt(
        llm: LanguageModel,
        schema: dict,
        prompt_builder: PromptBuilder = DefaultPromptBuilder(),
        response_parser: ResponseParser = DefaultResponseParser()
) -> Any:
    return _prompt_json_entity(llm, prompt_builder, response_parser, schema)

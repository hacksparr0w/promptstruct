import json

from typing import Any, Awaitable, Callable, TypedDict


__all__ = (
    "Complete",
    "JsonBuildError",
    "ObjectSchema",
    "Schema",
    "UnsupportedJsonType",

    "build_json"
)


class Schema(TypedDict):
    type: str | list[str]
    description: str


class ObjectSchema(Schema, total=False):
    properties: dict[str, Schema]


Complete = Callable[[Schema], Awaitable[str]]


class JsonBuildError(Exception):
    pass


class UnsupportedJsonType(JsonBuildError):
    pass


async def _build_json_value(
    type: str,
    nullable: bool,
    schema: Schema,
    complete: Complete
) -> Any:
    value = json.loads(await complete(schema))

    if nullable and value is None:
        return None

    match type:
        case "string":
            return str(value)
        case "number":
            return float(value)
        case "integer":
            return int(value)
        case "boolean":
            return bool(value)
        case "null":
            if value is None:
                return None

            raise ValueError

    raise RuntimeError


async def _build_json_object(
    schema: ObjectSchema,
    complete: Complete
) -> dict:
    properties = schema["properties"]
    result = {}

    for key, subschema in properties.items():
        result[key] = await _build_json_entity(subschema, complete)

    return result


async def _build_json_entity(
    schema: Schema,
    complete: Complete
) -> Any:
    type = schema["type"]
    nullable = False

    if isinstance(type, list):
        if len(type) == 1:
            type = type[0]
        elif len(type) == 2 and "null" in type:
            if "object" in type:
                raise UnsupportedJsonType(type)

            a, b = type

            if a == "null":
                type = b
                nullable = True
            elif b == "null":
                type = a
                nullable = True
        else:
            raise UnsupportedJsonType(type)

    match type:
        case "object":
            return await _build_json_object(schema, complete)
        case "string" | "number" | "integer" | "boolean" | "null":
            return await _build_json_value(type, nullable, schema, complete)

    raise UnsupportedJsonType(type)


async def build_json(schema: Schema, complete: Complete) -> Any:
    return await _build_json_entity(schema, complete)

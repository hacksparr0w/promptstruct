from textwrap import dedent

from openai import AsyncOpenAI

from ._json import Schema


__all__ = (
    "default_complete",
)


async def default_complete(
    schema: Schema,
    client: AsyncOpenAI
) -> str:
    type = schema["type"]
    description = schema["description"]

    completion = await client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": dedent(
                    """
                    You are a large language model tasked with generating a
                    JSON-compatible values by answering user's prompt.
                    Only respond with the value that best answers the user's
                    prompt, do not include any other information.

                    Following are some examples of inputs and outputs you
                    should try to replicate.

                    Prompt: The capital of the United States
                    Type: string
                    Answer: "Washington, D.C."

                    Prompt: Answer to the Ultimate Question of Life
                    Type: integer
                    Answer: 42

                    Prompt: What is the first number in the following
                    sequence? "a, b, c, d, e, f, g, h, i, j, k, l, m, n, ..."
                    Type: ['integer', 'null']
                    Answer: null
                    """
                )
            },
            {
                "role": "user",
                "content": dedent(
                    f"""
                    Prompt: {description}
                    Type: {type}
                    Answer:
                    """
                )
            }
        ],
        model="gpt-4-1106-preview"
    )

    return completion.choices[0].message.content

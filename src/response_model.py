"""Pydantic model example"""

import asyncio

from dotenv import load_dotenv
from pydantic_ai import Agent

load_dotenv()


async def main() -> None:
    """Main function"""
    agent = Agent("openai:gpt-4o-mini", result_type=bool)

    result = await agent.run(
        "ケンタッキーフライドチキンの「カーネル・サンダース」のメガネには度が入っていない。◯か✕か。",
    )

    print(f"result: {result.data}, data-type: {type(result.data)}")


if __name__ == "__main__":
    asyncio.run(main())

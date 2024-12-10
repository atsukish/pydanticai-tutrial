"""hello world"""

import asyncio

from dotenv import load_dotenv
from pydantic_ai import Agent

load_dotenv()


async def main() -> None:
    """Hello world by pydanticai"""
    agent = Agent(
        "openai:gpt-4o-mini",
        system_prompt=("すべてツンデレ口調で回答してください。"),
    )

    result = await agent.run("東ティモールの首都は？")
    print(result.data)


if __name__ == "__main__":
    asyncio.run(main())











"""Pydantic model example"""

import asyncio

from pydantic import BaseModel
from pydantic_ai import Agent


class WorldCupInfo(BaseModel):
    """city location"""

    year: int
    """year"""
    host_country: str
    """host country name"""
    winner: str | None
    """winner country name"""


async def main() -> None:
    """Main function"""
    agent = Agent("openai:gpt-4o-mini", result_type=list[WorldCupInfo])

    result = await agent.run(
        "1990年から2026年までのサッカーワールドカップの開催国と優勝国を列挙してください。",
    )

    for data in result.data:
        print(data.model_dump())


if __name__ == "__main__":
    asyncio.run(main())

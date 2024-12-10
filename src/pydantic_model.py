"""Pydantic model example"""

import asyncio

from pydantic import BaseModel
from pydantic_ai import Agent


class CityLocation(BaseModel):
    """city location"""

    city: str
    """city name"""
    country: str
    """country name"""


async def main() -> None:
    """Main function"""
    agent = Agent("openai:gpt-4o-mini", result_type=CityLocation)

    result = await agent.run("2020年のオリンピック開催地は？")

    print(result.data.model_dump())
    print(result.cost())


if __name__ == "__main__":
    asyncio.run(main())

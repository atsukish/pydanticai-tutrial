"""system prompt with dependencies"""

import asyncio
import os
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from tabulate import tabulate

load_dotenv()


@dataclass
class NewsAPIDeps:
    """Dependencies for NewsAPI"""

    api_key: str
    """API key for NewsAPI"""
    http_client: httpx.AsyncClient
    """HTTP client"""


agent = Agent(
    "openai:gpt-4o",
    deps_type=NewsAPIDeps,
)


@agent.system_prompt
async def get_system_prompt(ctx: RunContext[NewsAPIDeps]) -> str:
    """Get system prompt"""
    headers = {"X-Api-Key": ctx.deps.api_key}
    response = await ctx.deps.http_client.get(
        url="https://newsapi.org/v2/everything",
        headers=httpx.Headers(headers),
        params={
            "q": "AI AND エージェント",
            "sortBy": "publishedAt",
            "pageSize": 10,
        },
    )
    response.raise_for_status()
    data = response.json()

    articles = [
        {
            "title": article["title"],
            "author": article["author"],
            "url": article["url"],
            "publishedAt": article["publishedAt"],
        }
        for article in data["articles"]
    ]

    return (
        f"あなたはAIの専門家です。今日のAIに関するニュースは以下のとおりです。\n\n"
        f"{tabulate(articles, headers='keys')}"
        f"\n\nこれらのニュースを参考にして、ユーザーと会話してください。"
        "なお、ニュースを引用する際は必ず出典元のURLを含めてください。"
    )


async def main() -> None:
    """Main function"""
    async with httpx.AsyncClient() as client:
        deps = NewsAPIDeps(
            api_key=os.environ["NEWS_API_KEY"],
            http_client=client,
        )
        result = await agent.run(
            "AIエージェントの最新動向とそれに対する考察をしてみて",
            deps=deps,
        )
        print(result.data)


if __name__ == "__main__":
    asyncio.run(main())

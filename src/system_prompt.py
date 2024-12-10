import asyncio
from datetime import date

from pydantic_ai import Agent, RunContext

agent = Agent(
    "openai:gpt-4o-mini",
    deps_type=str,
    system_prompt="ユーザーの名前を使って返信してください。",
)


@agent.system_prompt
def add_the_users_name(ctx: RunContext[str]) -> str:
    """Return the user's name"""
    return f"ユーザーの名前は {ctx.deps} です。"


@agent.system_prompt
def add_the_date() -> str:
    """Return today's date"""
    return f"今日の日付は {date.today()} です。"


async def main() -> None:
    """Main function"""
    result = await agent.run("今日の日付は？", deps="田中")
    print(result.data)
    print(result.all_messages())


if __name__ == "__main__":
    asyncio.run(main())

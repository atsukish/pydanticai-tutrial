import asyncio
import random

from pydantic_ai import Agent, RunContext

agent = Agent(
    "openai:gpt-4o-mini",
    deps_type=str,
    system_prompt=(
        "あなたはサイコロゲームです。サイコロを振って、出た目がユーザーの予想と"
        "一致するかどうかを確認してください。一致した場合は、プレイヤーの名前を使って"
        "勝者であることを伝えてください。"
    ),
)


@agent.tool_plain
def roll_die() -> str:
    """Roll a six-sided die and return the result."""
    return str(random.randint(1, 6))


@agent.tool
def get_player_name(ctx: RunContext[str]) -> str:
    """Get the player's name."""
    return ctx.deps


async def main() -> None:
    dice_result = await agent.run("私の予想は「4」です", deps="たかし")
    print(dice_result.data)
    print(dice_result.all_messages())


if __name__ == "__main__":
    asyncio.run(main())

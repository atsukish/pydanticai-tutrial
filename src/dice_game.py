import asyncio
import random

from pydantic_ai import Agent, RunContext, result

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
def roll_dice() -> str:
    """Roll a six-sided dice and return the result.

    Returns:
        str: The result of the die roll.
    """
    return str(random.randint(1, 6))


@agent.tool
def get_player_name(ctx: RunContext[str]) -> str:
    """Get the player's name.

    Args:
        ctx (RunContext[str]): context of the run(user's name).

    Returns:
        str: The player's name.
    """
    return ctx.deps


async def dice_game(username: str, prompt: str) -> result.RunResult[str]:
    """Dice game

    Args:
        username (str): The player's name.
        prompt (str): The player's guess.

    Returns:
        result.RunResult: The result of the dice game.
    """
    return await agent.run(prompt, deps=username)


async def main() -> None:
    """Main function"""
    dice_result = await dice_game(
        username="たかし",
        prompt="私の予想は「4」です",
    )
    print(dice_result.data)
    print(dice_result.all_messages())


if __name__ == "__main__":
    asyncio.run(main())

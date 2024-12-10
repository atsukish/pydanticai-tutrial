"""unit test dice game agent"""

from datetime import timezone

import pytest
from dirty_equals import IsNow
from pydantic_ai import models
from pydantic_ai.messages import (
    ArgsDict,
    ModelStructuredResponse,
    ModelTextResponse,
    SystemPrompt,
    ToolCall,
    ToolReturn,
    UserPrompt,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from dice_game import agent, dice_game

models.ALLOW_MODEL_REQUESTS = False


@pytest.mark.asyncio
async def test_dice_game() -> None:
    """Test dice game"""
    with agent.override(model=TestModel()):
        prompt = "私の予想は「4」です"
        username = "たかし"
        _ = await dice_game(username, prompt)

    assert len(agent.last_run_messages) == 6

    for index, message in enumerate(agent.last_run_messages):
        print(index, message)

    assert agent.last_run_messages[0] == SystemPrompt(
        content="あなたはサイコロゲームです。サイコロを振って、出た目がユーザーの予想と一致するかどうかを確認してください。一致した場合は、プレイヤーの名前を使って勝者であることを伝えてください。",
        role="system",
    )
    assert agent.last_run_messages[1] == UserPrompt(
        content=prompt,
        timestamp=IsNow(tz=timezone.utc),
        role="user",
    )
    assert agent.last_run_messages[2] == ModelStructuredResponse(
        calls=[
            ToolCall(
                tool_name="roll_dice",
                args=ArgsDict({}),
                tool_id=None,
            ),
            ToolCall(
                tool_name="get_player_name",
                args=ArgsDict({}),
                tool_id=None,
            ),
        ],
        timestamp=IsNow(tz=timezone.utc),
        role="model-structured-response",
    )
    # assert agent.last_run_messages[3] == ToolReturn(
    #     tool_name="get_player_name",
    #     content=username,
    #     timestamp=IsNow(tz=timezone.utc),
    #     role="tool-return",
    # )

    assert agent.last_run_messages[4] == ToolReturn(
        tool_name="get_player_name",
        content=username,
        timestamp=IsNow(tz=timezone.utc),
        role="tool-return",
    )

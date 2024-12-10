"""unit test sample"""

from datetime import date, timezone

import pytest
from dirty_equals import IsNow
from pydantic_ai import models
from pydantic_ai.messages import (
    ModelTextResponse,
    SystemPrompt,
    UserPrompt,
)
from pydantic_ai.models.test import TestModel

from system_prompt import agent

models.ALLOW_MODEL_REQUESTS = False


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_system_prompt_agent() -> None:
    """Test system prompt"""
    # モックモデルを利用
    with agent.override(model=TestModel(custom_result_text="モック回答だよ")):
        prompt = "おはよう"
        username = "松本"
        _ = await agent.run(prompt, deps=username)

    assert agent.last_run_messages == [
        SystemPrompt(
            content="ユーザーの名前を使って返信してください。",
            role="system",
        ),
        SystemPrompt(
            content=f"ユーザーの名前は {username} です。",
            role="system",
        ),
        SystemPrompt(
            content=f"今日の日付は {date.today()} です。",
            role="system",
        ),
        UserPrompt(
            content=prompt,
            timestamp=IsNow(tz=timezone.utc),
            role="user",
        ),
        ModelTextResponse(
            content="モック回答だよ",
            timestamp=IsNow(tz=timezone.utc),
            role="model-text-response",
        ),
    ]

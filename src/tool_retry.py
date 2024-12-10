from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext


class FakeDatabase:
    users = {"John Doe": 123, "Jane Smith": 456}


class ChatResult(BaseModel):
    user_id: int
    message: str


agent = Agent(
    "openai:gpt-4o-mini",
    deps_type=FakeDatabase,
    result_type=ChatResult,
)


@agent.tool(retries=2)
def get_user_by_name(ctx: RunContext[FakeDatabase], name: str) -> int:
    """Get a user's ID from their full name."""
    print(name)
    user_id = ctx.deps.users.get(name)
    if user_id is None:
        print(
            f"No user found with name {name!r}, remember to provide their full name",
        )
        raise ModelRetry(
            f"No user found with name {name!r}, remember to provide their full name",
        )
    return user_id


result = agent.run_sync(
    "Send a message to John asking for coffee next week",
    deps=FakeDatabase(),
)
print(result.data)
print(result.cost())
print(result.all_messages_json().decode("utf-8"))

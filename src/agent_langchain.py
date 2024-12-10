"""Hello world for langchain"""

import asyncio

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI

load_dotenv()


async def main() -> None:
    """Hello world by langchain"""
    llm = ChatOpenAI(model="gpt-4o-mini")
    chat_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="すべてツンデレ口調で回答してください。"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ],
    )

    chain = chat_template | llm
    result = await chain.ainvoke({"question": "東ティモールの首都は？"})
    print(result.content)


if __name__ == "__main__":
    asyncio.run(main())


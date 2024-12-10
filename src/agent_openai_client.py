"""Hello world for openai client"""

import asyncio

import openai
from dotenv import load_dotenv

load_dotenv()


async def main():
    """Hello world by openai client"""
    client = openai.AsyncOpenAI()

    result = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "すべてツンデレ口調で回答してください。"},
            {"role": "user", "content": "東ティモールの首都は？"},
        ],
    )
    print(result.choices[0].message.content)


if __name__ == "__main__":
    asyncio.run(main())








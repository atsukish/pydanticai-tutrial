"""Streamed user profile"""

import asyncio
from datetime import date

from pydantic_ai import Agent
from typing_extensions import TypedDict


class PlayerProfile(TypedDict, total=False):
    """Player profile"""

    name: str
    birth_date: date
    birth_place: str
    team: str
    position: str
    nicknamed: str


async def main():
    """Main function"""
    agent = Agent(
        "openai:gpt-4o",
        result_type=PlayerProfile,
        system_prompt="与えられた情報から選手のプロフィールを抽出してください。",
    )

    # 大谷翔平のwikipediaのページより
    user_input = (
        "大谷 翔平（おおたに しょうへい、1994年7月5日 - ）は、岩手県奥州市出身の"
        "プロ野球選手（投手、指名打者、外野手）。右投左打。MLBのロサンゼルス・ドジャース所属。"
        "多くの野球関係者から史上最高の野球選手の1人として評価されている"
        "近代プロ野球では極めて稀なシーズンを通して投手と野手を兼任する「二刀流（英: two-way player）」の選手"
        "メジャーリーグベースボール（MLB）/日本プロ野球（NPB）両リーグで「1シーズンでの2桁勝利投手・2桁本塁打」を達成。"
        "NPBで最優秀選手を1度受賞、MLBでシーズンMVP（最優秀選手賞）を3度受賞。"
        "近代MLBにおいて同一年に規定投球回数と規定打席数の両方に到達した史上初の選手"
        "MLBにおいて日本人初、アジア人初の本塁打王と打点王獲得者。"
    )
    async with agent.run_stream(user_input) as result:
        async for profile in result.stream():
            print(profile)


if __name__ == "__main__":
    asyncio.run(main())

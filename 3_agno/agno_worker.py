"""Agno 워커를 별도의 subprocess로 실행하여 보드(board) 작업을 처리한다.

인자 없이 실행하면 자체 Day 3 목표를 직접 생성하고 수행한다
(notes.txt 읽기 -> 스페인어 번역 -> spanish.txt 쓰기):
보드에 단계를 계획하고, 파일시스템 MCP 서버를 통해 파일을 읽고,
번역한 뒤 스페인어 결과를 기록하고, 각 단계를 완료 처리한 후 목표를 닫는다.

task id와 공유 보드 경로를 인자로 주면 같은 워커가 Day 5 에이전트 루프에 합류한다:
보드 도구와 파일 도구를 공유 보드 및 공유 사이트로 향하게 하고,
해당 task 하나만 claim하여 요구된 작업을 수행한 뒤 종료하며, 보드의 나머지는 건드리지 않는다.

    uv run agno_worker.py                       # 단독 실행 Day 3 데모
    uv run agno_worker.py <taskId> <boardPath>  # Day 5: 공유 보드에서 task 하나를 처리
"""

from __future__ import annotations

import asyncio
import functools
import os
import subprocess
import sys
from pathlib import Path

# Day 5 모드의 인자 형식은 "<taskId> <boardPath>"이다.
# board 모듈은 import 시점에 BOARD_PATH 환경변수로 대상 파일을 결정하므로,
# board를 import 하기 전에 인자를 먼저 읽어 환경변수를 설정한다.
# 인자 없이 실행하면 아래 블록은 동작하지 않는다.
TASK_ID = int(sys.argv[1]) if len(sys.argv) > 2 else None
if TASK_ID is not None:
    # 이미 BOARD_PATH가 설정되어 있으면 덮어쓰지 않는다.
    os.environ.setdefault("BOARD_PATH", sys.argv[2])

from dotenv import load_dotenv  # noqa: E402
from agno.agent import Agent  # noqa: E402
from agno.models.openai import OpenAIChat  # noqa: E402
from agno.tools.mcp import MCPTools  # noqa: E402
from mcp import StdioServerParameters  # noqa: E402

# Agno의 MCP 도구는 서버의 stderr를 외부로 노출하지 않으므로 stdio 클라이언트의
# errlog를 DEVNULL로 지정한다. 파일시스템 서버의 시작 배너 출력이 사라지고,
# Windows의 Jupyter 커널에서도 실행이 가능해진다.
import agno.tools.mcp.mcp as agno_mcp  # noqa: E402

# functools.partial로 stdio_client 호출 시 errlog 인자를 기본으로 고정한다.
agno_mcp.stdio_client = functools.partial(agno_mcp.stdio_client, errlog=subprocess.DEVNULL)

import board  # noqa: E402

# .env 파일을 읽어 환경변수에 적용한다. override=True이므로 기존 값도 덮어쓴다.
load_dotenv(override=True)

# 워커가 사용할 모델 이름. 환경변수 WORKER_MODEL이 없으면 기본값을 사용한다.
MODEL = os.environ.get("WORKER_MODEL", "gpt-5.4-mini")

# 이 스크립트 파일이 있는 디렉터리 하위의 workspace 폴더 경로
WORKSPACE = Path(__file__).resolve().parent / "workspace"

# 단독 실행 모드에서 수행할 목표 문장 (한글)
GOAL = "notes.txt를 읽고, 그 내용을 자연스러운 스페인어로 번역한 뒤, 그 스페인어를 spanish.txt에 작성하라."

# 파일 도구가 쓰기를 수행할 디렉터리를 결정한다.
# 단독 실행이면 워커 자신의 workspace, Day 5 모드면 공유 보드 파일이 위치한 폴더를 사용한다.
WORK_DIR = WORKSPACE if TASK_ID is None else Path(sys.argv[2]).resolve().parent

# 지정한 id의 OpenAI 채팅 모델 객체를 생성한다.
model = OpenAIChat(id=MODEL)


def show_todos() -> list[dict]:
    """보드에 있는 모든 todo 항목을 나열한다. 목표(goal)는 parent_id가 None이고, 단계(step)는 parent_id에 해당 목표의 id가 들어 있다."""
    # board 모듈에서 전체 todo 목록을 가져와 그대로 반환한다.
    return board.list_todos()


def plan_steps(goal_id: int, steps: list[str]) -> dict:
    """하나의 목표를 순서가 있는 단계 체크리스트로 나누어 보드에 등록한다. 목표의 id와 짧은 단계 설명 목록을 전달한다."""
    # 각 단계 설명을 순서대로 보드에 추가하고, 생성된 단계 id 목록을 함께 반환한다.
    return {"goal_id": goal_id, "step_ids": [board.add_step(goal_id, step) for step in steps]}


def complete_task(task_id: int, result: str) -> dict:
    """이 id를 가진 todo(단계 또는 목표)를 완료(done) 상태로 표시하고 짧은 결과 요약을 기록한다."""
    # 보드에서 해당 항목을 완료 처리하고 결과 문자열을 저장한다.
    board.complete_todo(task_id, result)
    return {"task_id": task_id, "status": "done"}


# 에이전트에게 전달되는 시스템 지시문 (한글)
INSTRUCTIONS = """
당신은 공유 todo 보드와 파일 도구 모음을 사용하는 신중한 작업자입니다.

대기 중인(pending) 목표를 맡아 끝까지 완수하십시오. 먼저 짧은 계획을 세우는 것으로 시작합니다:
해당 작업이 실제로 나뉘는 몇 개의 구체적인 단계를 정리하여, 그 목표 아래 보드에 추가하십시오.
그런 다음 파일 도구를 사용해 각 단계를 수행하고, 하나를 끝낼 때마다 그 단계를 완료로 표시하십시오.
모든 단계가 완료되면 목표를 닫으십시오. 당신의 파일은 도구가 사용 허가를 받은 단 하나의 폴더 안에 있습니다.
"""


def seed() -> int:
    """보드를 초기화하고, 이전 출력물을 지운 뒤, 목표 하나를 추가한다."""
    # 보드의 기존 내용을 모두 비운다.
    board.reset_board()
    # workspace 폴더가 없으면 생성한다.
    WORKSPACE.mkdir(exist_ok=True)
    # 이전 실행에서 생성된 결과 파일이 있으면 삭제한다. 없으면 무시한다.
    (WORKSPACE / "spanish.txt").unlink(missing_ok=True)
    # 목표를 보드에 추가하고 그 id를 받는다.
    goal_id = board.add_goal(GOAL)
    # 워커가 목표를 가져간다: 상태가 pending -> in_progress 로 바뀐다.
    board.claim_todo(goal_id)
    return goal_id


async def main() -> None:
    if TASK_ID is None:
        # 단독 실행 모드: 보드를 직접 준비하고 목표를 생성한다.
        goal_id = seed()
        print(f"목표 {goal_id} 생성 완료: {GOAL}\n")
        # 에이전트에게 전달할 입력 메시지 (한글)
        message = "보드에 있는 대기 중인 목표를 처리해 주세요."
    else:
        # Day 5 모드: 공유 보드에서 지정된 task 하나만 claim 한다.
        board.claim_todo(TASK_ID)
        # 에이전트에게 전달할 입력 메시지 (한글)
        message = (
            f"당신은 공유 보드에서 task #{TASK_ID} 를 맡았습니다. 오직 그 task와 그 하위 단계만 처리하십시오. "
            f"작업을 완성하고 확인까지 마쳤다면 complete_task 로 task #{TASK_ID} 자체를 완료 표시한 뒤 중단하십시오."
        )

    # 파일시스템 MCP 서버를 stdio 방식으로 실행하기 위한 파라미터를 구성한다.
    # 서버에는 WORK_DIR 한 곳만 접근 허용 경로로 전달한다.
    server = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(WORK_DIR)],
        cwd=str(WORK_DIR),
    )
    # MCP 서버에 연결하고, 컨텍스트를 벗어나면 자동으로 연결을 종료한다.
    async with MCPTools(server_params=server, timeout_seconds=60) as filesystem:
        # 모델, 지시문, 도구 목록(보드 도구 3개 + 파일시스템 MCP 도구)을 묶어 에이전트를 만든다.
        worker = Agent(
            model=model,
            instructions=INSTRUCTIONS,
            tools=[show_todos, plan_steps, complete_task, filesystem],
        )
        # 준비된 메시지를 입력으로 에이전트를 비동기 실행한다.
        await worker.arun(input=message)

    # 단독 실행 모드에서만 결과를 출력한다. Day 5 모드에서는 오케스트레이터가 콘솔 출력을 담당한다.
    if TASK_ID is None:
        print("\n실행 후 보드 상태:")
        board.show_board()
        spanish = WORKSPACE / "spanish.txt"
        # 결과 파일이 생성되었으면 그 내용을 UTF-8로 읽어 출력한다.
        if spanish.exists():
            print("\nspanish.txt:\n" + spanish.read_text(encoding="utf-8"))


if __name__ == "__main__":
    # 비동기 main 코루틴을 이벤트 루프에서 실행한다.
    asyncio.run(main())
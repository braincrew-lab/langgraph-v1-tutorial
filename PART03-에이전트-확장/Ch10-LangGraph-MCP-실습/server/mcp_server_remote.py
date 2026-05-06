from mcp.server.fastmcp import FastMCP
from typing import Optional
import pytz
from datetime import datetime

# FastMCP 서버 초기화 및 구성
mcp = FastMCP(
    "Current Time",  # MCP 서버 이름
    instructions="주어진 시간대의 현재 시간 정보를 제공합니다",
    port=8002,  # HTTP 전송 시 사용할 포트
)


@mcp.tool()
async def get_current_time(timezone: Optional[str] = "Asia/Seoul") -> str:
    """지정된 시간대의 현재 시간 정보를 가져옵니다.

    이 함수는 요청된 시간대의 현재 시스템 시간을 반환합니다.

    Args:
        timezone: 현재 시간을 조회할 시간대. 기본값은 "Asia/Seoul"입니다.

    Returns:
        지정된 시간대의 현재 시간 정보가 포함된 문자열
    """
    try:
        # 시간대 객체를 가져옵니다
        tz = pytz.timezone(timezone)

        # 지정된 시간대의 현재 시간을 가져옵니다
        current_time = datetime.now(tz)

        # 시간을 문자열로 포맷합니다
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S %Z")

        return f"Current time in {timezone} is: {formatted_time}"
    except pytz.exceptions.UnknownTimeZoneError:
        return f"Error: Unknown timezone '{timezone}'. Please provide a valid timezone."
    except Exception as e:
        return f"Error getting time: {str(e)}"


if __name__ == "__main__":
    # 서버가 시작됨을 알리는 메시지를 출력합니다
    print("MCP Remote 서버가 실행 중입니다...")

    # streamable-http 전송 방식으로 서버를 시작합니다 (포트 8002)
    mcp.run(transport="streamable-http")

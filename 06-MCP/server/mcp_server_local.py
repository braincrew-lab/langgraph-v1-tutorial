from mcp.server.fastmcp import FastMCP

# FastMCP 서버 초기화 및 구성
mcp = FastMCP(
    "Weather",  # MCP 서버 이름
    instructions="날씨 정보를 제공하는 어시스턴트입니다. 주어진 위치의 날씨에 대한 질문에 답변할 수 있습니다.",
)


@mcp.tool()
async def get_weather(location: str) -> str:
    """지정된 위치의 현재 날씨 정보를 가져옵니다.

    이 함수는 날씨 서비스를 시뮬레이션하여 고정된 응답을 반환합니다.
    실제 프로덕션 환경에서는 실제 날씨 API에 연결해야 합니다.

    Args:
        location: 날씨를 조회할 위치의 이름 (도시, 지역 등)

    Returns:
        지정된 위치의 날씨 정보가 포함된 문자열
    """
    # 모의 날씨 응답을 반환합니다
    # 실제 구현에서는 날씨 API를 호출해야 합니다
    return f"It's always Sunny in {location}"


if __name__ == "__main__":
    # stdio 전송 방식으로 MCP 서버를 시작합니다
    # stdio 전송은 표준 입출력 스트림을 통해 클라이언트와 통신하며,
    # 로컬 개발 및 테스트에 적합합니다
    mcp.run(transport="stdio")

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_teddynote.tools.tavily import TavilySearch

load_dotenv(override=True)

# 모델 초기화
model = init_chat_model("claude-sonnet-4-5")

# 도구 정의: Tavily 검색 도구
tools = [TavilySearch(max_results=3)]

# 에이전트 그래프 생성
# langgraph dev 서버가 체크포인터를 자동으로 관리하므로 별도 설정 불필요
graph = create_agent(model, tools=tools)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from typing import Any
from rag.pdf import PDFRetrievalChain


load_dotenv(override=True)


def create_retriever() -> Any:
    """PDF 문서에서 Retriever를 생성합니다.

    Returns:
        PDF 문서 기반의 Retriever 객체
    """
    import os

    # 현재 디렉토리 기준으로 PDF 파일 경로를 설정합니다
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_dir, "data", "SPRI_AI_Brief_2023년12월호_F.pdf")

    # PDF 문서를 로드하고 체인을 생성합니다
    pdf = PDFRetrievalChain([pdf_path]).create_chain()

    # retriever를 반환합니다
    pdf_retriever = pdf.retriever

    return pdf_retriever


# FastMCP 서버 초기화 및 구성
mcp = FastMCP(
    "Retriever",
    instructions="데이터베이스에서 정보를 검색할 수 있는 Retriever입니다.",
)


@mcp.tool()
async def retrieve(query: str) -> str:
    """쿼리를 기반으로 문서 데이터베이스에서 정보를 검색합니다.

    이 함수는 Retriever를 생성하고, 제공된 쿼리로 검색을 수행한 후,
    검색된 모든 문서의 내용을 연결하여 반환합니다.

    Args:
        query: 관련 정보를 찾기 위한 검색 쿼리

    Returns:
        검색된 모든 문서의 텍스트 내용을 연결한 문자열
    """
    retriever = create_retriever()

    # invoke() 메서드를 사용하여 쿼리 기반의 관련 문서를 가져옵니다
    retrieved_docs = retriever.invoke(query)

    # 모든 문서 내용을 줄바꿈으로 연결하여 단일 문자열로 반환합니다
    return "\n".join([doc.page_content for doc in retrieved_docs])


if __name__ == "__main__":
    # MCP 클라이언트와의 통합을 위해 stdio 전송 방식으로 서버를 실행합니다
    mcp.run(transport="stdio")

from mcp.server.fastmcp import FastMCP
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_tavily import TavilySearch
from langchain_core.documents import Document
from dotenv import load_dotenv
from typing import List, Literal
import os


load_dotenv(override=True)

# FastMCP 서버 초기화
mcp = FastMCP(
    "RAG_Server",
    instructions="벡터 검색, 문서 추가, 웹 검색 기능을 제공하는 RAG 서버입니다."
)

# 전역 변수로 벡터스토어 관리
vector_store = None
embeddings = OpenAIEmbeddings()


def initialize_vector_store():
    """벡터 스토어를 초기화하고 PDF 문서를 로드합니다.

    PDF 문서를 청크로 분할하고 FAISS 벡터 스토어에 저장합니다.

    Returns:
        초기화된 FAISS 벡터 스토어
    """
    global vector_store

    # 현재 디렉토리 기준으로 PDF 파일 경로를 설정합니다
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_dir, "data", "SPRI_AI_Brief_2023년12월호_F.pdf")

    # PDF 문서를 로드합니다
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    # 문서를 청크로 분할합니다
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)

    # FAISS 벡터 스토어를 생성합니다
    vector_store = FAISS.from_documents(splits, embeddings)
    return vector_store


@mcp.tool()
async def vector_search(
    query: str,
    search_type: Literal["semantic", "keyword", "hybrid"] = "semantic",
    k: int = 5
) -> str:
    """벡터 스토어에서 문서를 검색합니다.

    Args:
        query: 검색 쿼리
        search_type: 검색 유형 ("semantic", "keyword", "hybrid")
        k: 반환할 결과 수

    Returns:
        검색된 문서 내용
    """
    global vector_store

    # 벡터 스토어가 초기화되지 않았으면 초기화합니다
    if vector_store is None:
        initialize_vector_store()

    # 검색 유형에 따라 다른 검색 방식을 사용합니다
    if search_type == "semantic":
        # 시맨틱 검색: 의미적 유사성 기반
        results = vector_store.similarity_search(query, k=k)
    elif search_type == "keyword":
        # 키워드 검색: 문자열 매칭 기반
        all_docs = vector_store.similarity_search("", k=100)
        results = [doc for doc in all_docs if query.lower() in doc.page_content.lower()][:k]
    elif search_type == "hybrid":
        # 하이브리드 검색: 시맨틱 + 키워드 결합
        semantic_results = vector_store.similarity_search(query, k=k*2)
        keyword_results = [doc for doc in semantic_results if query.lower() in doc.page_content.lower()]
        results = keyword_results[:k] if keyword_results else semantic_results[:k]

    return "\n\n".join([doc.page_content for doc in results])


@mcp.tool()
async def add_document(text: str, metadata: dict = None) -> str:
    """사용자 텍스트를 벡터 스토어에 추가합니다.

    Args:
        text: 추가할 텍스트 내용
        metadata: 문서 메타데이터 (선택적)

    Returns:
        추가 결과 메시지
    """
    global vector_store

    # 벡터 스토어가 초기화되지 않았으면 초기화합니다
    if vector_store is None:
        initialize_vector_store()

    # 메타데이터가 없으면 기본값을 설정합니다
    if metadata is None:
        metadata = {"source": "user_input"}

    # 텍스트를 청크로 분할합니다
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    # 문서 객체를 생성하고 분할합니다
    documents = [Document(page_content=text, metadata=metadata)]
    splits = text_splitter.split_documents(documents)

    # 벡터 스토어에 문서를 추가합니다
    vector_store.add_documents(splits)

    return f"문서가 성공적으로 추가되었습니다. 총 {len(text)} 문자, {len(splits)}개 청크로 분할됨"


@mcp.tool()
async def web_search(query: str, max_results: int = 3) -> str:
    """TavilySearch를 사용하여 웹 검색을 수행합니다.

    Args:
        query: 검색 쿼리
        max_results: 최대 결과 수

    Returns:
        포맷된 검색 결과
    """
    # Tavily 검색 클라이언트를 생성합니다
    tavily = TavilySearch(max_results=max_results)
    results = tavily.invoke(query)

    # 검색 결과를 포맷합니다
    formatted_results = []
    for i, result in enumerate(results, 1):
        formatted_results.append(
            f"검색 결과 {i}:\n"
            f"제목: {result.get('title', 'N/A')}\n"
            f"URL: {result.get('url', 'N/A')}\n"
            f"내용: {result.get('content', 'N/A')}\n"
        )

    return "\n".join(formatted_results)


if __name__ == "__main__":
    # 서버 초기화
    print("RAG MCP 서버를 초기화합니다...")
    initialize_vector_store()
    print("벡터 스토어 초기화 완료!")

    # MCP 서버 실행 (stdio 전송 방식)
    mcp.run(transport="stdio")

from .base import RetrievalChain
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Annotated
from pathlib import Path
import os
import hashlib


class PDFRetrievalChain(RetrievalChain):
    def __init__(self, source_uri: Annotated[str, "Source URI"], **kwargs):
        super().__init__(**kwargs)
        self.source_uri = source_uri

        # PDF 파일 경로 기반으로 고유한 캐시 디렉토리 생성
        if isinstance(source_uri, str):
            file_hash = hashlib.md5(source_uri.encode()).hexdigest()[:8]
            file_name = Path(source_uri).stem
            cache_suffix = f"{file_name}_{file_hash}"

            self.cache_dir = Path(f".cache/embeddings/{cache_suffix}")
            self.index_dir = Path(f".cache/faiss_index/{cache_suffix}")
            print(f"Cache configured for PDF: {file_name}")
            print(f"- Embeddings cache: {self.cache_dir}")
            print(f"- FAISS index cache: {self.index_dir}")
        else:
            # 여러 파일의 경우 기본 캐시 사용
            self.cache_dir = Path(".cache/embeddings/multi_pdf")
            self.index_dir = Path(".cache/faiss_index/multi_pdf")
            print("Cache configured for multi-PDF processing")

    def load_documents(self, source_uris: List[str]) -> List[Document]:
        docs = []
        successful_files = 0
        failed_files = []

        for source_uri in source_uris:
            try:
                # 파일 존재 및 권한 확인
                file_path = Path(source_uri)
                if not file_path.exists():
                    print(f"Warning: File not found: {source_uri}")
                    failed_files.append(source_uri)
                    continue

                if not file_path.is_file():
                    print(f"Warning: Not a file: {source_uri}")
                    failed_files.append(source_uri)
                    continue

                if not os.access(source_uri, os.R_OK):
                    print(f"Warning: No read permission: {source_uri}")
                    failed_files.append(source_uri)
                    continue

                # PDF 파일 확장자 확인
                if not source_uri.lower().endswith(".pdf"):
                    print(f"Warning: Not a PDF file: {source_uri}")
                    failed_files.append(source_uri)
                    continue

                # PDF 로딩 시도
                print(f"Loading PDF: {source_uri}")
                loader = PDFPlumberLoader(source_uri)
                loaded_docs = loader.load()

                if not loaded_docs:
                    print(f"Warning: No content loaded from: {source_uri}")
                    failed_files.append(source_uri)
                    continue

                docs.extend(loaded_docs)
                successful_files += 1
                print(
                    f"Successfully loaded {len(loaded_docs)} pages from: {source_uri}"
                )

            except Exception as e:
                print(f"Error loading PDF {source_uri}: {e}")
                failed_files.append(source_uri)
                continue

        # 결과 요약 출력
        print(f"\nLoading Summary:")
        print(f"- Successfully loaded: {successful_files} files")
        print(f"- Failed to load: {len(failed_files)} files")
        if failed_files:
            print(f"- Failed files: {failed_files}")
        print(f"- Total documents loaded: {len(docs)}")

        if not docs:
            raise ValueError(
                "No documents were successfully loaded from the provided source URIs"
            )

        return docs

    def create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )

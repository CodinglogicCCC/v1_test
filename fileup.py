import os
import logging
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore  # 최신 PineconeVectorStore 사용
import tiktoken
from dotenv import load_dotenv



dotenv_path = os.path.join(os.getcwd(), '.env')
load_dotenv(dotenv_path)

# Pinecone API Key 확인
pinecone_api_key = os.getenv("PINECONE_API_KEY")
print(f"Pinecone API Key: {pinecone_api_key}")  


# 로깅 설정
logging.basicConfig(level=logging.INFO)

# ✅ Pinecone 초기화
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "v1-test"

# ✅ Pinecone 인덱스 연결
try:
    index = pc.Index(index_name)
    logging.info(f"✅ Pinecone 인덱스 '{index_name}'에 연결 성공!")
except Exception as e:
    logging.error(f"❌ Pinecone 인덱스 연결 실패: {e}")
    exit(1)  # 실행 중단

# ✅ Pinecone 인덱스 차원 확인 (확인용)
index_info = index.describe_index_stats()
print("📌 Pinecone 인덱스 메타데이터:", index_info)

# ✅ 기존 데이터 삭제
def delete_existing_pinecone_data():
    try:
        index.delete(delete_all=True)  # 전체 벡터 삭제
        logging.info("🗑 기존 Pinecone 데이터 삭제 완료!")
    except Exception as e:
        logging.error(f"❌ Pinecone 데이터 삭제 실패: {e}")

# ✅ 작은 단위로 텍스트 청크 분할
def split_text_into_chunks(text, max_tokens=2048, model="text-embedding-3-small"):
    tokenizer = tiktoken.encoding_for_model(model)
    tokens = tokenizer.encode(text)
    chunks = []
    current_chunk = []

    for token in tokens:
        current_chunk.append(token)
        if len(current_chunk) >= max_tokens:
            chunks.append(tokenizer.decode(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(tokenizer.decode(current_chunk))

    return chunks

# ✅ Pinecone에 새로운 Markdown 데이터 업로드
def upload_docs_to_pinecone(markdown_folder: str):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  # ✅ Pinecone 인덱스(`1536`)와 맞춤!
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # ✅ 임베딩 벡터 차원 확인 (테스트 실행)
    test_vector = embeddings.embed_query("테스트")
    print(f"🔍 생성된 벡터 차원: {len(test_vector)}")  # ✅ 1536이어야 정상

    try:
        vector_store = PineconeVectorStore(
            index=index, embedding=embeddings, text_key="text"
        )  # 최신 방식으로 Pinecone 연결
        logging.info("✅ PineconeVectorStore 초기화 성공!")
    except Exception as e:
        logging.error(f"❌ Pinecone VectorStore 초기화 실패: {e}")
        return

    # ✅ Markdown 파일 목록 가져오기
    all_docs = [file for file in os.listdir(markdown_folder) if file.endswith(".md")]

    if not all_docs:
        logging.warning("⚠️ 업로드할 문서를 찾을 수 없습니다!")
        return

    logging.info(f"📂 업로드할 파일 목록: {all_docs}")

    for file_name in all_docs:
        file_path = os.path.join(markdown_folder, file_name)
        if not os.path.exists(file_path):
            logging.warning(f"⚠️ 파일 없음: {file_name}, 업로드 생략")
            continue

        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

            # 🔹 "서울과학기술대학교" 문구 포함 여부에 따라 카테고리 설정
            category = "school-docs" if "서울과학기술대학교" in content else "policy-docs"

            # 🔹 Markdown 원본을 그대로 청크로 분할
            chunks = split_text_into_chunks(content, max_tokens=2048)  # 작은 단위로 조정

            for i, chunk in enumerate(chunks):
                try:
                    # ✅ 벡터 차원 확인 (업로드 직전)
                    vector = embeddings.embed_query(chunk[:100])  # 일부 텍스트만 벡터화
                    print(f"📌 {file_name} 청크 {i+1} 벡터 차원: {len(vector)}")  # ✅ 1536이어야 정상

                    vector_store.add_texts(
                        [chunk],
                        metadatas=[{
                            "filename": file_name,
                            "category": category,
                            "format": "markdown",
                            "original_md": chunk
                        }]
                    )
                    logging.info(f"✅ {file_name}의 청크 {i + 1} 업로드 완료 (카테고리: {category})")
                except Exception as e:
                    logging.error(f"❌ {file_name} 업로드 실패: {e}")

# ✅ 실행
if __name__ == "__main__":
    markdown_folder = "C:/Users/sukyo/privacychat/DB/v1 최초/MD"

    # ✅ 기존 Pinecone 데이터 삭제
    delete_existing_pinecone_data()

    # ✅ 새로운 문서 업로드
    upload_docs_to_pinecone(markdown_folder)

    print("🚀 문서 업로드 완료! (Markdown 원본 유지, 기존 데이터 삭제 후 새 데이터 반영)")

import os
import re
import logging
import traceback
import json
from typing import List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from dotenv import load_dotenv
from pydantic import Field

# .env 로드
load_dotenv()

# 세션별 대화 히스토리 저장소
store = {}

# 로깅 설정
logging.basicConfig(level=logging.INFO)

class CustomRetriever(BaseRetriever):
    """schooldocs에서 검색 후, 부족하면 policydocs에서 추가 검색하는 검색기"""

    schooldocs_retriever: BaseRetriever = Field(...)
    policydocs_retriever: BaseRetriever = Field(...)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """학교 문서를 우선 검색하고, 부족하면 법규 문서 검색 추가"""
        logging.info(f"검색 쿼리: {query}")
        
        # 1단계: schooldocs에서 검색
        school_docs = self.schooldocs_retriever.invoke(query)
        logging.info(f" schooldocs 검색 완료: {len(school_docs)}개 문서 반환됨")

        # 검색된 문서가 부족하면 policydocs 추가 검색
        if len(school_docs) < 3:
            policy_docs = self.policydocs_retriever.invoke(query)
            logging.info(f" policydocs 추가 검색 완료: {len(policy_docs)}개 문서 반환됨")
            return school_docs + policy_docs
        
        return school_docs


def clean_reference(text: str) -> str:
    """AI 응답에서 출처 정보를 명확하게 정리"""
    pattern = r'출처:\s*([^"\)]+)"?\s*\(([^\)]+)\)'
    matches = re.findall(pattern, text)
    
    for match in matches:
        doc_name, page_info = match[0].strip(), match[1]
        formatted_ref = f"(출처: 문서명: {doc_name}, 페이지: {page_info})"
        text = text.replace(f"출처: {match[0]} ({match[1]})", formatted_ref)
    
    return text


def get_retriever(category=None) -> PineconeVectorStore:
    """Pinecone 기반 벡터 검색기 생성 (필터 적용 가능)"""
    try:
        embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
        index_name = os.getenv("PINECONE_INDEX", "v1-test")  # 하나의 인덱스 사용
        database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)

        # 기본적으로 schooldocs에서 검색
        search_kwargs = {"k": 3}
        if category:
            search_kwargs["filter"] = {"category": category}  # 특정 카테고리만 검색하도록 필터 추가

        return database.as_retriever(search_kwargs=search_kwargs)
    except Exception as e:
        logging.error(f" get_retriever() 실패: {e}")
        return None


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """세션 ID별 대화 히스토리 관리"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_llm(model: str = "gpt-3.5-turbo") -> ChatOpenAI:
    """OpenAI LLM 인스턴스 생성"""
    return ChatOpenAI(model=model, streaming=True)


def get_rag_chain():
    """RAG 기반 응답 생성 (schooldocs 우선 검색, 필요 시 policydocs 추가 검색)"""
    try:
        logging.info(" get_rag_chain() 실행 시작")

        llm = get_llm()
        if not llm:
            logging.error("LLM 생성 실패로 인해 RAG 체인을 만들 수 없습니다.")
            return None

        #  schooldocs와 policydocs 검색기 생성
        retriever_school = get_retriever(category="schooldocs")
        retriever_policy = get_retriever(category="policydocs")

        if not retriever_school or not retriever_policy:
            logging.error(" 검색기 생성 실패")
            return None

        # Combined Retriever
        combined_retriever = CustomRetriever(
            schooldocs_retriever=retriever_school, 
            policydocs_retriever=retriever_policy
        )

        #  QA 프롬프트 설정
        system_prompt = (
            "당신은 개인정보 보호 전문가입니다. 제공된 문서를 활용하여 답변하세요. "
            "출처는 '(출처: 문서명: [문서명], 페이지: [페이지])' 형식으로 명확하게 표시하세요.\n\n{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        logging.info(" QA 프롬프트 설정 완료")

        #  Retrieval Chain 생성
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(combined_retriever, question_answer_chain)

        logging.info(" RAG 체인 생성 완료")

        return RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input", history_messages_key="chat_history", output_messages_key="answer"
        )

    except Exception as e:
        logging.error(f"get_rag_chain() 오류 발생: {e}")
        logging.error(traceback.format_exc())
        return None


def get_ai_response(user_message: str, session_id: str):
    """AI 응답을 스트리밍하여 반환하고 JSON 파일에 저장"""
    try:
        logging.info(f" 사용자 질문: {user_message}")
        rag_chain = get_rag_chain()

        if not rag_chain:
            logging.error("get_rag_chain() 반환값이 None입니다.")
            return ["현재 AI 응답을 생성할 수 없습니다."]

        response = []
        logging.info("AI 응답 생성 시작...")

        for chunk in rag_chain.stream(
            {"input": user_message},
            config={"configurable": {"session_id": session_id}}
        ):
            response.append(chunk.get("answer", ""))

        final_response = "".join(response)
        logging.info(f"최종 AI 응답: {final_response}")

        # AI 응답을 JSON 파일에 저장
        output_file = "ai_responses.json"
        response_data = {
            "session_id": session_id,
            "user_message": user_message,
            "ai_response": final_response
        }

        # 기존 응답이 있다면 불러와서 덧붙임
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as file:
                existing_data = json.load(file)
        else:
            existing_data = []

        existing_data.append(response_data)

        # 응답 데이터를 JSON 파일에 저장
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(existing_data, file, ensure_ascii=False, indent=4)

        return final_response

    except Exception as e:
        logging.error(f" AI 응답 생성 중 오류 발생: {e}")
        logging.error(traceback.format_exc())
        return [" AI 응답 생성 중 오류가 발생했습니다."]
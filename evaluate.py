import os
import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import json

# NLTK 다운로드 (최초 실행 시 필요)
nltk.download("punkt")

# 📌 평가 질문 및 정답 리스트
test_questions = [
    "서울과기대가 수집하는 정보 뭐 있어?",
    "내 개인정보 언제까지 보관해?",
    "학교에서 내 정보를 어디에 써?",
    "졸업하면 내 개인정보 어떻게 돼?",
    "학교에서 내 정보 삭제할 수 있어?"
]

reference_answers = {
    "서울과기대가 수집하는 정보 뭐 있어?": "서울과학기술대학교는 이름, 연락처, 이메일을 수집합니다.",
    "내 개인정보 언제까지 보관해?": "개인정보는 5년간 보관됩니다.",
    "학교에서 내 정보를 어디에 써?": "학교는 학사 관리, 성적 처리, 행정 업무에 개인정보를 사용합니다.",
    "졸업하면 내 개인정보 어떻게 돼?": "졸업 후에도 학적 정보는 일정 기간 동안 보관됩니다.",
    "학교에서 내 정보 삭제할 수 있어?": "법적으로 보관해야 하는 정보는 삭제되지 않습니다."
}

# 📌 평가 결과 저장할 리스트
results = []

# 📌 평가 함수
def evaluate_bleu_rouge(reference, hypothesis):
    smoothie = SmoothingFunction().method4
    bleu = sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=smoothie)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge = scorer.score(reference, hypothesis)
    
    return {
        "BLEU": round(bleu, 4),
        "ROUGE-1": round(rouge["rouge1"].fmeasure, 4),
        "ROUGE-2": round(rouge["rouge2"].fmeasure, 4),
        "ROUGE-L": round(rouge["rougeL"].fmeasure, 4)
    }

# 📌 평가 실행 함수
def run_evaluation(version, chatbot_responses):
    print(f"\n🔍 버전 {version} 평가 시작...")

    for question in test_questions:
        # ai_responses.json에서 user_message와 ai_response를 가져옴
        chatbot_response = chatbot_responses.get(question, {}).get("ai_response", "응답 없음")
        reference = reference_answers.get(question, "")

        scores = evaluate_bleu_rouge(reference, chatbot_response)

        results.append({
            "버전": version,
            "질문": question,
            "챗봇 응답": chatbot_response,
            "BLEU": scores["BLEU"],
            "ROUGE-1": scores["ROUGE-1"],
            "ROUGE-2": scores["ROUGE-2"],
            "ROUGE-L": scores["ROUGE-L"]
        })

    print(f"✅ 버전 {version} 평가 완료!")

# 📌 버전별 응답 불러오기
def load_chatbot_responses(version):
    file_path = f"ai_responses.json"  # 파일 이름을 ai_responses.json으로 수정

    if not os.path.exists(file_path):
        print(f"⚠ {file_path} 파일이 없습니다. 먼저 챗봇을 실행하고 응답을 저장하세요.")
        return {}

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        if isinstance(data, list):
            # 리스트 형태로 되어있으면 딕셔너리 형태로 변경 (각 질문을 키로, 응답을 값으로)
            return {item["user_message"]: item for item in data}  # 'user_message'와 'ai_response'로 변경
        return {}

# 📌 버전별 평가 실행
for version in ["v1"]:
    responses = load_chatbot_responses(version)
    if responses:
        run_evaluation(version, responses)

# 📌 결과를 CSV로 저장
df = pd.DataFrame(results)
df.to_csv("evaluation_results_v1.csv", index=False, encoding="utf-8-sig")

print("\n📊 모든 평가 완료! 결과가 'evaluation_results_v1.csv' 파일에 저장되었습니다.")

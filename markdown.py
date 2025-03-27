import os
import fitz  # PyMuPDF

def convert_pdf_to_markdown(pdf_folder: str, output_folder: str):
    """PDF 파일을 Markdown으로 변환하여 저장"""
    
    # 변환된 Markdown을 저장할 폴더가 없으면 생성
    os.makedirs(output_folder, exist_ok=True)

    # PDF 폴더에서 파일 읽기
    for file_name in os.listdir(pdf_folder):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, file_name)
            markdown_path = os.path.join(output_folder, file_name.replace(".pdf", ".md"))

            print(f"📄 변환 중: {pdf_path} → {markdown_path}")

            # PDF 파일 열기
            doc = fitz.open(pdf_path)
            markdown_content = ""

            # 페이지별 텍스트 추출
            for page_num in range(len(doc)):
                text = doc[page_num].get_text("text")
                if text:
                    markdown_content += f"## Page {page_num + 1}\n\n{text}\n\n"

            # Markdown 파일 저장
            with open(markdown_path, "w", encoding="utf-8") as md_file:
                md_file.write(markdown_content)
            
            print(f"✅ 변환 완료: {markdown_path}")

# 실행 예시
pdf_folder = "C:/Users/sukyo/privacychat/DB/v1 최초"
output_folder = "C:/Users/sukyo/privacychat/DB/v1 최초/새 폴더더"
convert_pdf_to_markdown(pdf_folder, output_folder)

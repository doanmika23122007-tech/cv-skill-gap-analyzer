import re
import json
import pdfplumber
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# 1. CẤU HÌNH GEMINI API KEY (Điền API Key của bạn vào đây)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 2. PIPELINE ĐỌC VÀ LÀM SẠCH CV
# ---------------------------------------------------------------------------
def clean_and_sanitize_text(raw_text: str) -> str:
    cleaned = re.sub(r'[\uE000-\uF8FF]', '', raw_text)
    cleaned = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL_MASKED]', cleaned)
    cleaned = re.sub(r'(\b0|\+84)\d{8,9}\b', '[PHONE_MASKED]', cleaned)
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    lines = [line.strip() for line in cleaned.splitlines()]
    return "\n".join([line for line in lines if line])

def extract_text_from_pdf(pdf_path: str) -> str:
    full_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text(layout=True)
                if text:
                    full_text.append(text)
        raw_content = "\n".join(full_text)
        if not raw_content.strip():
            return ""
        return clean_and_sanitize_text(raw_content)
    except Exception as e:
        print(f"Lỗi đọc PDF: {e}")
        return ""

# ---------------------------------------------------------------------------
# 3. AI ENGINE: PHÂN TÍCH SKILL GAP QUA GEMINI API
# ---------------------------------------------------------------------------
def analyze_skill_gap(cv_text: str, jd_text: str) -> dict:
    """
    Sử dụng Gemini API để so sánh CV và JD, trả về kết quả dạng JSON.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""
    Bạn là một chuyên gia tuyển dụng và đánh giá nhân sự AI/Data Science hàng đầu.
    Nhiệm vụ của bạn là phân tích CV của ứng viên và Mô tả công việc (JD), sau đó đưa ra đánh giá chính xác.

    --- NỘI DUNG CV CỦA ỨNG VIÊN ---
    {cv_text}

    --- NỘI DUNG MÔ TẢ CÔNG VIỆC (JD) ---
    {jd_text}

    --- YÊU CẦU ĐẦU RA ---
    Hãy trả về duy nhất 1 chuỗi định dạng JSON chuẩn (không chứa markdown ```json) với cấu trúc sau:
    {{
        "match_score": <điểm_phù_hợp_từ_0_đến_100>,
        "matched_skills": [<danh_sách_kỹ_năng_mà_ứng_viên_ĐÃ_CÓ_đáp_ứng_được_JD>],
        "missing_skills": [<danh_sách_kỹ_năng_JD_yêu_cầu_nhưng_CV_ĐANG_THIẾU>],
        "summary_evaluation": "<đánh_giá_ngắn_gọn_2-3_câu_về_khả_năng_trúng_tuyển>",
        "actionable_advice": [<3_gợi_ý_cụ_thể_để_ứng_viên_bổ_sung_hồ_sơ_trong_2-4_tuần>]
    }}
    """

    response = client.models.generate_content(
        model='gemini-flash-latest',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json" # Ép Gemini trả về dạng JSON chuẩn
        )
    )

    try:
        # Chuyển chuỗi JSON từ AI thành Dict trong Python
        return json.loads(response.text)
    except Exception as e:
        print(f"Lỗi parse JSON từ AI: {e}")
        return {"raw_response": response.text}

# ---------------------------------------------------------------------------
# 4. CHẠY THỬ NGHIỆM
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_pdf = "sample_cv.pdf"
    
    # 1. Lấy CV sạch
    cv_clean = extract_text_from_pdf(sample_pdf)
    
    # 2. Tạo một JD mẫu để thử nghiệm
    sample_jd = """
    VỊ TRÍ: Thực tập sinh AI / Data Science (AI/DS Intern)
    Yêu cầu công việc:
    - Đang là sinh viên ngành Công nghệ thông tin, AI, Khoa học dữ liệu.
    - Thành thạo ngôn ngữ lập trình Python và các thư viện xử lý dữ liệu (Pandas, NumPy).
    - Có kiến thức cơ bản về Machine Learning, SQL / Hệ cơ sở dữ liệu.
    - Biết sử dụng Git/GitHub để quản lý mã nguồn.
    - Ưu tiên ứng viên có kinh nghiệm làm việc với NLP, LLM API hoặc thư viện OpenCV / PyTorch.
    - Tiếng Anh đọc hiểu tài liệu kỹ thuật tốt.
    """

    print("🤖 Đang gửi dữ liệu sang Gemini AI để phân tích...")
    analysis_result = analyze_skill_gap(cv_clean, sample_jd)
    
    print("\n=== KẾT QUẢ PHÂN TÍCH TỪ AI (SKILL-GAP REPORT) ===")
    print(json.dumps(analysis_result, ensure_ascii=False, indent=4))
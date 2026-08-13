import streamlit as st
import re
import json
import time
import pdfplumber
from google import genai
from google.genai import types
from google.genai.errors import ServerError, APIError

# ---------------------------------------------------------------------------
# 1. CẤU HÌNH TRANG STREAMLIT
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CV Skill-Gap Analyzer (Hybrid Engine)",
    page_icon="🛡️",
    layout="wide"
)

# ---------------------------------------------------------------------------
# 2. DICTIONARY TỪ KHÓA CÔNG NGHỆ CHUẨN (RULE-BASED TAXONOMY)
# ---------------------------------------------------------------------------
TECH_KEYWORDS = [
    # Ngôn ngữ lập trình
    "Python", "C\+\+", "Java", "JavaScript", "TypeScript", "SQL", "R", "HTML", "CSS",
    # Thư viện Data & AI
    "Pandas", "NumPy", "Matplotlib", "Seaborn", "Scikit-Learn", "PyTorch", "TensorFlow", 
    "Keras", "OpenCV", "MediaPipe", "NLTK", "Spacy", "Transformers", "FastAPI", "Flask", "Django",
    # Công cụ & Hạ tầng
    "Git", "GitHub", "Docker", "Kubernetes", "Jupyter", "VS Code", "PostgreSQL", "MySQL", 
    "MongoDB", "Streamlit", "Gradio", "Linux", "AWS", "GCP", "Azure"
]

# ---------------------------------------------------------------------------
# 3. TẦNG 1: BỘ LỌC ĐỐI SOÁT TỪ KHÓA BẰNG PYTHON REGEX
# ---------------------------------------------------------------------------
def extract_hard_skills_with_regex(text: str) -> set:
    found_skills = set()
    for skill in TECH_KEYWORDS:
        pattern = r'(?i)\b' + skill + r'\b'
        if re.search(pattern, text):
            clean_name = skill.replace("\\", "")
            found_skills.add(clean_name)
    return found_skills

# ---------------------------------------------------------------------------
# 4. TẦNG 2: CLEANING & PIPELINE ĐỌC CV
# ---------------------------------------------------------------------------
def clean_and_sanitize_text(raw_text: str) -> str:
    cleaned = re.sub(r'[\uE000-\uF8FF]', '', raw_text)
    cleaned = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL_MASKED]', cleaned)
    cleaned = re.sub(r'(\b0|\+84)\d{8,9}\b', '[PHONE_MASKED]', cleaned)
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    lines = [line.strip() for line in cleaned.splitlines()]
    return "\n".join([line for line in lines if line])

def extract_text_from_pdf_file(uploaded_file) -> str:
    full_text = []
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text(layout=True)
                if text:
                    full_text.append(text)
        raw_content = "\n".join(full_text)
        if not raw_content.strip():
            return ""
        return clean_and_sanitize_text(raw_content)
    except Exception as e:
        st.error(f"Lỗi khi đọc file PDF: {e}")
        return ""

# ---------------------------------------------------------------------------
# 5. TẦNG 3: GEMINI AI ENGINE (CÓ CƠ CHẾ KHẮC PHỤC NGHẼN NETWORK / 503 RETRY)
# ---------------------------------------------------------------------------
def analyze_skill_gap_hybrid(api_key: str, cv_text: str, jd_text: str, py_cv_skills: set, py_jd_skills: set) -> dict:
    client = genai.Client(api_key=api_key)

    py_matched = list(py_cv_skills.intersection(py_jd_skills))
    py_missing = list(py_jd_skills - py_cv_skills)

    prompt = f"""
    Bạn là hệ thống kiểm định CV chuyên nghiệp. Hãy phân tích CV và JD dưới đây.

    --- DỮ LIỆU ĐỐI SOÁT TỪ MÃ PYTHON (ĐÃ XÁC THỰC 100% BẰNG REGEX) ---
    - Các từ khóa kỹ thuật cứng CV VÀ JD ĐỀU CÓ: {py_matched}
    - Các từ khóa kỹ thuật cứng JD CẦN NHƯNG CV THIẾU: {py_missing}

    --- NỘI DUNG CV ---
    {cv_text}

    --- NỘI DUNG JD ---
    {jd_text}

    --- YÊU CẦU ---
    Hãy kết hợp dữ liệu đối soát từ Python và phân tích ngữ cảnh để trả về DUY NHẤT 1 chuỗi JSON chuẩn:
    {{
        "match_score": <chữ_số_từ_0_đến_100>,
        "matched_skills": [<danh_sách_kỹ_năng_CV_đáp_ứng_bao_gồm_cả_kỹ_năng_cứng_và_mềm>],
        "missing_skills": [<danh_sách_kỹ_năng_còn_thiếu_so_với_JD>],
        "summary_evaluation": "<đánh_giá_khách_quan_2-3_câu>",
        "actionable_advice": [<3_gợi_ý_bổ_sung_hồ_sơ_thực_tế>]
    }}
    """

    # Danh sách các tên model dự phòng nếu một model bị nghẽn (503)
    candidate_models = ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-flash-latest']
    last_exception = None

    for model_name in candidate_models:
        for attempt in range(2):  # Thử lại tối đa 2 lần cho mỗi model
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.0
                    )
                )
                return json.loads(response.text)
            except ServerError as e:
                last_exception = e
                time.sleep(2)  # Chờ 2 giây nếu nghẽn mạng rồi tự động thử lại
            except Exception as e:
                last_exception = e
                break  # Nếu lỗi khác thì chuyển model ngay

    raise last_exception

# ---------------------------------------------------------------------------
# 6. GIAO DIỆN STREAMLIT UI
# ---------------------------------------------------------------------------
st.title("🛡️ AI Skill-Gap Analyzer (Hybrid Engine)")
st.caption("Hệ thống phân tích khoảng cách kỹ năng đa tầng: Python Regex + Gemini AI Agent")

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Cấu hình Hệ thống")
    api_key_input = st.text_input("Nhập Gemini API Key:", type="password")
    st.markdown("---")
    st.markdown("**Kiến trúc Đa tầng:**\n- 🐍 **Tầng 1:** Python Regex Hard-skill Matcher\n- 🤖 **Tầng 2:** Gemini LLM Semantic Analyzer")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 1. Tải lên CV (PDF)")
    uploaded_cv = st.file_uploader("Chọn file CV dạng PDF", type=["pdf"])

with col2:
    st.subheader("📋 2. Nhập Job Description (JD)")
    jd_input = st.text_area("Dán nội dung tuyển dụng/JD vào đây:", height=200)

st.markdown("---")
if st.button("🔍 Phân Tích CV Ngay (Hybrid Check)", type="primary", use_container_width=True):
    if not api_key_input:
        st.warning("⚠️ Vui lòng nhập Gemini API Key!")
    elif not uploaded_cv:
        st.warning("⚠️ Vui lòng tải lên file PDF CV!")
    elif not jd_input.strip():
        st.warning("⚠️ Vui lòng dán nội dung JD!")
    else:
        with st.spinner("⚡ Đang chạy Kiểm tra Đa tầng (Python Regex + Gemini AI)..."):
            cv_text = extract_text_from_pdf_file(uploaded_cv)
            
            if not cv_text:
                st.error("❌ File PDF không chứa dữ liệu văn bản!")
            else:
                # 1. Chạy Tầng 1 (Python Regex)
                py_cv_skills = extract_hard_skills_with_regex(cv_text)
                py_jd_skills = extract_hard_skills_with_regex(jd_input)
                
                # 2. Chạy Tầng 2 & 3 (AI + Hybrid Verification + Retry Logic)
                try:
                    result = analyze_skill_gap_hybrid(api_key_input, cv_text, jd_input, py_cv_skills, py_jd_skills)
                    
                    # --- HIỂN THỊ KẾT QUẢ ---
                    st.success("✅ Phân tích Đa tầng Hoàn tất!")
                    
                    # Hiển thị đối soát nhanh Tầng 1
                    with st.expander("🐍 Xem Đối soát Từ khóa Cứng độc lập bởi Python Regex (Tầng 1)", expanded=True):
                        p_col1, p_col2 = st.columns(2)
                        with p_col1:
                            st.write("**Từ khóa Kỹ thuật CV & JD ĐỀU CÓ:**")
                            st.info(", ".join(py_cv_skills.intersection(py_jd_skills)) if py_cv_skills.intersection(py_jd_skills) else "Không tìm thấy từ khóa khớp trực tiếp")
                        with p_col2:
                            st.write("**Từ khóa Kỹ thuật JD CẦN nhưng CV THIẾU:**")
                            st.error(", ".join(py_jd_skills - py_cv_skills) if (py_jd_skills - py_cv_skills) else "Không thiếu từ khóa kỹ thuật cứng nào!")

                    st.markdown("---")
                    
                    # Hiển thị báo cáo AI Tầng 2
                    score = result.get("match_score", 0)
                    st.metric(label="🎯 Mức độ Phù hợp Tổng thể (Hybrid Match Score)", value=f"{score} / 100")
                    st.progress(score / 100)
                    
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.subheader("✅ Kỹ năng ĐÃ CÓ (Tổng hợp)")
                        for item in result.get("matched_skills", []):
                            st.write(f"• {item}")
                            
                    with res_col2:
                        st.subheader("❌ Kỹ năng ĐANG THIẾU (Tổng hợp)")
                        for item in result.get("missing_skills", []):
                            st.write(f"• {item}")
                    
                    st.markdown("---")
                    st.subheader("📝 Đánh giá Tổng quan")
                    st.info(result.get("summary_evaluation", ""))
                    
                    st.subheader("💡 Lời khuyên Hành động (Action Plan)")
                    for advice in result.get("actionable_advice", []):
                        st.success(f"👉 {advice}")

                except ServerError:
                    st.error("⏳ Máy chủ Google AI hiện đang quá tải lượt truy cập (Lỗi 503). Vui lòng đợi khoảng 5 - 10 giây rồi nhấn nút 'Phân tích' lại nhé!")
                except Exception as e:
                    st.error(f"❌ Có lỗi xảy ra: {str(e)}")
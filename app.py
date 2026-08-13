import streamlit as st
import re
import json
import time
import os
import pdfplumber
from google import genai
from google.genai import types
from google.genai.errors import ServerError

# ---------------------------------------------------------------------------
# 1. CẤU HÌNH TRANG STREAMLIT
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CV Skill-Gap & Job Matching Engine",
    page_icon="💼",
    layout="wide"
)

TECH_KEYWORDS = [
    "Python", "C\+\+", "Java", "JavaScript", "TypeScript", "SQL", "R", "HTML", "CSS",
    "Pandas", "NumPy", "Matplotlib", "Seaborn", "Scikit-Learn", "PyTorch", "TensorFlow", 
    "Keras", "OpenCV", "MediaPipe", "NLTK", "Spacy", "Transformers", "FastAPI", "Flask", "Django",
    "Git", "GitHub", "Docker", "Kubernetes", "Jupyter", "VS Code", "PostgreSQL", "MySQL", 
    "MongoDB", "Streamlit", "Gradio", "Linux", "AWS", "GCP", "Azure", "Machine Learning", "Deep Learning", "LLM"
]

# ---------------------------------------------------------------------------
# 2. HÀM ĐỌC DATABASE & CHUYỂN ĐỔI DỮ LIỆU
# ---------------------------------------------------------------------------
def load_jobs_database() -> list:
    if os.path.exists("jobs.json"):
        with open("jobs.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def extract_hard_skills_with_regex(text: str) -> set:
    found_skills = set()
    for skill in TECH_KEYWORDS:
        pattern = r'(?i)\b' + skill + r'\b'
        if re.search(pattern, text):
            clean_name = skill.replace("\\", "")
            found_skills.add(clean_name)
    return found_skills

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
# 3. ALGORITHM: LỌC TOP JOB KHỚP NHẤT VỚI CV
# ---------------------------------------------------------------------------
def find_top_matching_jobs(cv_skills: set, jobs_db: list, top_k: int = 3) -> list:
    scored_jobs = []
    for job in jobs_db:
        job_skills = set(job.get("skills", []))
        matched = cv_skills.intersection(job_skills)
        score = round((len(matched) / len(job_skills) * 100)) if job_skills else 0
        
        scored_jobs.append({
            "job_data": job,
            "match_score": score,
            "matched_skills": list(matched),
            "missing_skills": list(job_skills - cv_skills)
        })
    
    # Sắp xếp theo điểm số từ cao xuống thấp
    scored_jobs.sort(key=lambda x: x["match_score"], reverse=True)
    return scored_jobs[:top_k]

# ---------------------------------------------------------------------------
# 4. AI ENGINE: GEMINI TỔNG HỢP & ĐÁNH GIÁ CHI TIẾT
# ---------------------------------------------------------------------------
def evaluate_job_recommendations_with_ai(api_key: str, cv_text: str, top_jobs: list) -> dict:
    client = genai.Client(api_key=api_key)

    prompt = f"""
    Bạn là chuyên gia tư vấn định hướng nghề nghiệp AI/IT.
    Dưới đây là CV của ứng viên và Danh sách 3 công việc phù hợp nhất được hệ thống tìm thấy.

    --- CV CỦA ỨNG VIÊN ---
    {cv_text}

    --- TOP 3 CÔNG VIỆC TÌM THẤY ---
    {json.dumps(top_jobs, ensure_ascii=False, indent=2)}

    --- YÊU CẦU ---
    Hãy đưa ra đánh giá tổng quan ngắn gọn về độ tương thích công việc của ứng viên dưới dạng JSON chuẩn:
    {{
        "career_advice": "<Đánh giá 2-3 câu về vị trí công việc nào phù hợp nhất với hồ sơ hiện tại và hướng phát triển tiếp theo>",
        "top_recommendations": [
            {{
                "company": "<Tên công ty>",
                "title": "<Tên vị trí>",
                "reason": "<Lý do 1-2 câu vì sao ứng viên nên ứng tuyển công ty này>"
            }}
        ]
    }}
    """

    candidate_models = ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-flash-latest']
    for model_name in candidate_models:
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
        except Exception:
            continue
            
    return {"career_advice": "Không thể kết nối AI để sinh đánh giá chi tiết.", "top_recommendations": []}

# ---------------------------------------------------------------------------
# 5. GIAO DIỆN NGUỜI DÙNG (STREAMLIT UI)
# ---------------------------------------------------------------------------
st.title("💼 AI Job Matching & Skill-Gap Engine")
st.caption("Tự động tìm kiếm Công ty & Vị trí tuyển dụng phù hợp nhất với năng lực CV của bạn")

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Cấu hình Hệ thống")
    api_key_input = st.text_input("Nhập Gemini API Key:", type="password")
    
    jobs_db = load_jobs_database()
    st.success(f"📊 Đã nạp thành công **{len(jobs_db)}** tin tuyển dụng thực tế!")

# KHU VỰC UPLOAD CV
st.subheader("📄 Tải lên CV (PDF) của bạn để tìm công việc phù hợp")
uploaded_cv = st.file_uploader("Chọn file CV dạng PDF", type=["pdf"])

if st.button("🚀 Tìm Công Ty & Việc Làm Phù Hợp Ngay", type="primary", use_container_width=True):
    if not api_key_input:
        st.warning("⚠️ Vui lòng nhập Gemini API Key ở thanh bên trái!")
    elif not uploaded_cv:
        st.warning("⚠️ Vui lòng tải lên file PDF CV!")
    else:
        with st.spinner("🔍 Hệ thống đang trích xuất CV và quét đối soát với Cơ sở dữ liệu việc làm..."):
            cv_text = extract_text_from_pdf_file(uploaded_cv)
            
            if not cv_text:
                st.error("❌ Không thể đọc văn bản từ PDF này.")
            else:
                # 1. Trích xuất kỹ năng bằng Regex
                cv_skills = extract_hard_skills_with_regex(cv_text)
                
                # 2. Thuật toán lọc Top 3 công việc hợp nhất
                top_3_matches = find_top_matching_jobs(cv_skills, jobs_db, top_k=3)
                
                # 3. Gọi Gemini AI đánh giá tổng quan
                ai_evaluation = evaluate_job_recommendations_with_ai(api_key_input, cv_text, top_3_matches)
                
                st.success("🎉 Đã tìm thấy danh sách công ty và công việc phù hợp nhất với bạn!")
                
                # --- ĐÁNH GIÁ NGHỀ NGHIỆP TỪ AI ---
                st.markdown("---")
                st.subheader("💡 Tóm tắt Định hướng Nghề nghiệp")
                st.info(ai_evaluation.get("career_advice", ""))
                
                # --- HIỂN THỊ CÁC THẺ CÔNG VIỆC TOP MATCHING ---
                st.markdown("---")
                st.subheader("🏢 TOP 3 CÔNG TY & VỊ TRÍ PHÙ HỢP NHẤT")
                
                for idx, match in enumerate(top_3_matches, 1):
                    job = match["job_data"]
                    score = match["match_score"]
                    
                    with st.container():
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            st.markdown(f"### {idx}. {job['title']} — **{job['company']}**")
                            st.caption(f"📍 **Địa điểm:** {job['location']} | 🔗 [Trang tuyển dụng chính thức]({job['apply_link']})")
                            st.write(f"**Mô tả công việc:** {job['description']}")
                        with c2:
                            st.metric(label="Mức độ Khớp CV", value=f"{score}%")
                        
                        # Hiển thị kỹ năng đã có vs thiếu cho từng Job
                        sc1, sc2 = st.columns(2)
                        with sc1:
                            st.write("✅ **Kỹ năng CV đáp ứng:** " + (", ".join(match["matched_skills"]) if match["matched_skills"] else "Chưa ghi nhận"))
                        with sc2:
                            st.write("❌ **Kỹ năng cần bổ sung thêm:** " + (", ".join(match["missing_skills"]) if match["missing_skills"] else "Đã đáp ứng đủ!"))
                        
                        st.markdown("---")
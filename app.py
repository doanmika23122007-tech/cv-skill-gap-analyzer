import streamlit as st
import re
import json
import time
import os
import pdfplumber
import numpy as np
import unicodedata
from fpdf import FPDF
from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError, APIError

# ---------------------------------------------------------------------------
# 1. CẤU HÌNH TRANG STREAMLIT
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Job Matching & CV Optimizer Engine",
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
# 2. HÀM ĐỌC DATABASE & XỬ LÝ VĂN BẢN
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
# 3. TRỤC 1: OPTIMIZED VECTOR SEARCH (O(1) API CALL COMPLEXITY)
# ---------------------------------------------------------------------------
def get_text_embedding(api_key: str, text: str) -> list:
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=text
        )
        return response.embeddings[0].values
    except Exception:
        return None

def calculate_cosine_similarity(vec1: list, vec2: list) -> float:
    u = np.array(vec1)
    v = np.array(vec2)
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return float(dot_product / (norm_u * norm_v))

def find_top_matching_jobs_optimized(api_key: str, cv_text: str, cv_skills: set, jobs_db: list, top_k: int = 3) -> tuple:
    """
    Tối ưu O(1) API Call: Chỉ tạo Vector 1 lần duy nhất cho CV người dùng.
    """
    # 1. Tạo Vector cho CV
    cv_vector = get_text_embedding(api_key, cv_text)
    
    if cv_vector is not None:
        scored_jobs = []
        for job in jobs_db:
            # Lấy vector có sẵn của Job, nếu chưa có mới tính bổ sung
            if "embedding" in job and job["embedding"]:
                job_vector = job["embedding"]
            else:
                job_full_text = f"{job['title']} {job['company']} {job['description']} " + " ".join(job.get('skills', []))
                job_vector = get_text_embedding(api_key, job_full_text)
                job["embedding"] = job_vector # Cache lại
                
            if job_vector:
                semantic_score = calculate_cosine_similarity(cv_vector, job_vector)
                match_score = round(max(0, semantic_score) * 100)
            else:
                match_score = 0
                
            job_skills = set(job.get("skills", []))
            matched = cv_skills.intersection(job_skills)
            missing = list(job_skills - cv_skills)
            
            scored_jobs.append({
                "job_data": job,
                "match_score": match_score,
                "matched_skills": list(matched),
                "missing_skills": missing
            })
        
        scored_jobs.sort(key=lambda x: x["match_score"], reverse=True)
        return scored_jobs[:top_k], "Fast Vector Embeddings O(1)"

    # Fallback sang Regex nếu API Embedding gặp sự cố
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
    
    scored_jobs.sort(key=lambda x: x["match_score"], reverse=True)
    return scored_jobs[:top_k], "Python Regex Keyword Matcher (Fallback)"

def evaluate_job_recommendations_with_ai(api_key: str, cv_text: str, top_jobs: list) -> dict:
    client = genai.Client(api_key=api_key)
    prompt = f"""
    Bạn là chuyên gia tư vấn định hướng nghề nghiệp AI/IT.
    Dưới đây là CV ứng viên và Top 3 công việc phù hợp nhất.

    --- CV CỦA ỨNG VIÊN ---
    {cv_text}

    --- TOP 3 CÔNG VIỆC TÌM THẤY ---
    {json.dumps(top_jobs, ensure_ascii=False, indent=2)}

    --- YÊU CẦU ---
    Trả về JSON chuẩn:
    {{
        "career_advice": "<Đánh giá 2-3 câu về vị trí công việc phù hợp nhất>",
        "top_recommendations": [
            {{
                "company": "<Tên công ty>",
                "title": "<Tên vị trí>",
                "reason": "<Lý do vì sao ứng viên nên ứng tuyển>"
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
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0)
            )
            return json.loads(response.text)
        except Exception:
            continue
    return {"career_advice": "Không thể kết nối AI để sinh đánh giá chi tiết.", "top_recommendations": []}

def optimize_cv_bullet_points(api_key: str, raw_bullet: str) -> dict:
    client = genai.Client(api_key=api_key)
    prompt = f"""
    Bạn là một chuyên gia viết CV và tối ưu hồ sơ ứng tuyển ngành IT/AI.
    Hãy lấy câu mô tả kinh nghiệm/dự án sơ sài dưới đây của ứng viên và viết lại thành 3 phiên bản xuất sắc theo mô hình STAR (Situation - Task - Action - Result).

    --- MÔ TẢ CŨ CỦA ỨNG VIÊN ---
    "{raw_bullet}"

    --- YÊU CẦU ---
    Trả về DUY NHẤT 1 chuỗi JSON chuẩn:
    {{
        "optimized_bullets": [
            "Phiên bản 1 (Tập trung Kỹ thuật & Công nghệ)",
            "Phiên bản 2 (Tập trung Kết quả & Con số định lượng)",
            "Phiên bản 3 (Tập trung Tư duy Giải quyết Bài toán)"
        ],
        "key_keywords_added": ["Từ khóa công nghệ 1", "Từ khóa công nghệ 2"]
    }}
    """
    candidate_models = ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-flash-latest']
    for model_name in candidate_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2)
            )
            return json.loads(response.text)
        except Exception:
            continue
    return {"optimized_bullets": [], "key_keywords_added": []}

# ---------------------------------------------------------------------------
# 4. TRỤC 2: LEVEL 3 - PDF REPORT EXPORTER UTILITY
# ---------------------------------------------------------------------------
def clean_text_for_pdf(input_str: str) -> str:
    """Làm sạch triệt để Unicode (dấu tiếng Việt & ký tự đặc biệt từ AI) cho FPDF"""
    # 1. Thay thế các ký tự Unicode đặc biệt phổ biến mà Gemini hay sinh ra
    replacements = {
        '—': '-', '–': '-', '“': '"', '”': '"', '‘': "'", '’': "'",
        '•': '*', '…': '...', '–': '-', '—': '-'
    }
    for orig, repl in replacements.items():
        input_str = input_str.replace(orig, repl)
    
    # 2. Khử dấu tiếng Việt bằng NFKD
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    only_ascii = only_ascii.replace('đ', 'd').replace('Đ', 'D')
    
    # 3. Ép về ASCII thuần túy (bỏ mọi ký tự lạ còn sót)
    return only_ascii.encode('ascii', 'ignore').decode('ascii')

def generate_pdf_report(advice: str, top_matches: list) -> bytes:
    """Tạo file PDF báo cáo kết quả phân tích công việc"""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Title
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, txt="AI JOB MATCHING & SKILL-GAP REPORT", ln=1, align='C')
    pdf.ln(5)
    
    # Advice Section
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 8, txt="1. CAREER ADVICE SUMMARY:", ln=1)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 6, txt=clean_text_for_pdf(advice))
    pdf.ln(5)
    
    # Top Jobs Section
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 8, txt="2. TOP RECOMMENDED COMPANIES & JOBS:", ln=1)
    
    for idx, match in enumerate(top_matches, 1):
        job = match["job_data"]
        pdf.set_font("Helvetica", 'B', 10)
        pdf.cell(0, 6, txt=clean_text_for_pdf(f"{idx}. {job['title']} - {job['company']} (Match: {match['match_score']}%)"), ln=1)
        pdf.set_font("Helvetica", size=9)
        pdf.cell(0, 5, txt=clean_text_for_pdf(f"   Location: {job['location']}"), ln=1)
        pdf.multi_cell(0, 5, txt=clean_text_for_pdf(f"   Matched Skills: {', '.join(match['matched_skills'])}"))
        pdf.multi_cell(0, 5, txt=clean_text_for_pdf(f"   Missing Skills: {', '.join(match['missing_skills'])}"))
        pdf.ln(3)
        
    return bytes(pdf.output())

# ---------------------------------------------------------------------------
# 5. GIAO DIỆN STREAMLIT UI (CÓ TABS)
# ---------------------------------------------------------------------------
st.title("💼 AI Job Matching & CV Optimizer Engine")
st.caption("Hệ thống Phân tích Khớp nối Việc làm & Tối ưu Hồ sơ Chuẩn STAR")

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Cấu hình Hệ thống")
    api_key_input = st.text_input("Nhập Gemini API Key:", type="password")
    jobs_db = load_jobs_database()
    st.success(f"📊 Đã nạp thành công **{len(jobs_db)}** tin tuyển dụng thực tế!")

# TẠO TABS
tab1, tab2 = st.tabs(["🔍 1. Tìm Công Ty & Phân Tích Job", "✨ 2. AI Tối Ưu CV Chuẩn STAR"])

# --- TAB 1: JOB MATCHING ---
with tab1:
    st.subheader("📄 Tải lên CV (PDF) của bạn để tìm công việc phù hợp")
    uploaded_cv = st.file_uploader("Chọn file CV dạng PDF", type=["pdf"])

    if st.button("🚀 Tìm Công Ty & Việc Làm Phù Hợp Ngay", type="primary", use_container_width=True):
        if not api_key_input:
            st.warning("⚠️ Vui lòng nhập Gemini API Key ở thanh bên trái!")
        elif not uploaded_cv:
            st.warning("⚠️ Vui lòng tải lên file PDF CV!")
        else:
            with st.spinner("⚡ Đang tính toán Vector Embeddings O(1) & Lọc nhanh công việc..."):
                cv_text = extract_text_from_pdf_file(uploaded_cv)
                if not cv_text:
                    st.error("❌ Không thể đọc văn bản từ PDF này.")
                else:
                    cv_skills = extract_hard_skills_with_regex(cv_text)
                    
                    top_3_matches, engine_used = find_top_matching_jobs_optimized(api_key_input, cv_text, cv_skills, jobs_db, top_k=3)
                    ai_evaluation = evaluate_job_recommendations_with_ai(api_key_input, cv_text, top_3_matches)
                    
                    st.success(f"🎉 Hoàn tất phân tích! (Thuật toán: **{engine_used}**)")
                    
                    st.markdown("---")
                    st.subheader("💡 Tóm tắt Định hướng Nghề nghiệp")
                    advice_text = ai_evaluation.get("career_advice", "")
                    st.info(advice_text)
                    
                    # NÚT XUẤT BÁO CÁO PDF (LEVEL 3)
                    try:
                        pdf_bytes = generate_pdf_report(advice_text, top_3_matches)
                        st.download_button(
                            label="📥 Tải Báo Cáo Phân Tích (File PDF)",
                            data=pdf_bytes,
                            file_name="AI_Job_Matching_Report.pdf",
                            mime="application/pdf",
                            type="secondary"
                        )
                    except Exception as pdf_err:
                        st.caption(f"Không thể tạo file PDF preview: {pdf_err}")
                    
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
                            
                            sc1, sc2 = st.columns(2)
                            with sc1:
                                st.write("✅ **Kỹ năng CV đáp ứng:** " + (", ".join(match["matched_skills"]) if match["matched_skills"] else "Chưa ghi nhận"))
                            with sc2:
                                st.write("❌ **Kỹ năng cần bổ sung thêm:** " + (", ".join(match["missing_skills"]) if match["missing_skills"] else "Đã đáp ứng đủ!"))
                            
                            st.markdown("---")

# --- TAB 2: AI CV OPTIMIZER ---
with tab2:
    st.subheader("✨ Biến câu mô tả CV sơ sài thành Bullet-Points đắt giá chuẩn STAR")
    st.write("Nhập một dòng mô tả dự án hoặc kinh nghiệm làm việc cũ của bạn vào đây:")
    
    user_bullet_input = st.text_area(
        "Nhập câu mô tả cũ:",
        height=100,
        placeholder="Ví dụ: Tôi có làm một dự án phân tích dữ liệu CV bằng Python và Streamlit."
    )
    
    if st.button("🚀 Viết Lại Chuẩn STAR Ngay", type="primary"):
        if not api_key_input:
            st.warning("⚠️ Vui lòng nhập Gemini API Key ở thanh bên trái!")
        elif not user_bullet_input.strip():
            st.warning("⚠️ Vui lòng nhập câu mô tả cũ cần viết lại!")
        else:
            with st.spinner("🤖 AI đang phân tích ngữ cảnh và viết lại theo mô hình STAR..."):
                opt_result = optimize_cv_bullet_points(api_key_input, user_bullet_input)
                
                st.success("🎉 Đã tạo thành công các phiên bản CV chuẩn ATS!")
                
                st.markdown("### 📝 Các gợi ý viết lại đắt giá:")
                for idx, bullet in enumerate(opt_result.get("optimized_bullets", []), 1):
                    st.info(f"**Gợi ý {idx}:** {bullet}")
                
                keywords = opt_result.get("key_keywords_added", [])
                if keywords:
                    st.write("🔑 **Từ khóa công nghệ đắt giá được bổ sung:** " + ", ".join(keywords))
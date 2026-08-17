# 💼 AI Job Matching & CV Optimizer Engine

Hệ thống phân tích độ tương thích CV với thị trường tuyển dụng công nghệ tại Việt Nam và tối ưu hóa hồ sơ ứng tuyển theo chuẩn STAR bằng Gemini API & Vector Search.

🔗 **Live Demo:** [cv-skill.streamlit.app](https://cv-skill.streamlit.app)

---

## 🌟 Điểm Nổi Bật & Tính Năng Cốt Lõi

1. **Hybrid Job Matching Engine:**
   - **Semantic Search ($O(1)$ Embeddings):** So khớp ngữ nghĩa giữa CV và cơ sở dữ liệu việc làm bằng Vector Embeddings (`text-embedding-004`), tối ưu hóa chi phí và tốc độ truy vấn.
   - **Rule-based Fallback:** Tự động chuyển đổi sang bộ lọc Regex khi gặp sự cố mạng hoặc lỗi hạn ngạch API.

2. **AI CV Optimizer (Chuẩn STAR):**
   - Chuyển đổi mô tả công việc/dự án sơ sài thành 3 phiên bản chuyên nghiệp theo mô hình **STAR** (Situation - Task - Action - Result), bổ sung định lượng kết quả và từ khóa công nghệ đắt giá.

3. **Vietnam Tech Jobs Database:**
   - Quản lý cơ sở dữ liệu các tập đoàn công nghệ và trung tâm R&D hàng đầu tại Việt Nam (VinAI, Viettel AI, FPT Software, VNG, MoMo, Techcombank...).
   - Hỗ trợ xem trực tiếp kho JD trên giao diện web không cần upload CV.

4. **Báo Cáo Phân Tích PDF:**
   - Tự động xuất báo cáo đánh giá định hướng nghề nghiệp và danh sách việc làm phù hợp dưới dạng file PDF chuẩn hóa ký tự Unicode.

---

## 🛠️ Công Nghệ Sử Dụng (Tech Stack)

- **Frontend & UI:** Streamlit
- **AI & Embedding Engine:** Google GenAI SDK (`gemini-flash-latest`, `text-embedding-004`)
- **PDF Processing & Export:** `pdfplumber`, `fpdf2`
- **Data & Vector Computation:** `NumPy`, `Regex`, JSON

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Cục Bộ (Local Setup)

1. **Clone repository về máy:**
   ```bash
   git clone [https://github.com/doanmika23122007-tech/cv-skill-gap-analyzer.git](https://github.com/doanmika23122007-tech/cv-skill-gap-analyzer.git)
   cd cv-skill-gap-analyzer
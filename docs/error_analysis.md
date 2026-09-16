# Error Analysis Report — Vietnamese Job Skill NER

> **Tập đánh giá:** Test set (54 câu, 74 thực thể kỹ năng nhãn Ground Truth)  
> **Model checkpoint:** `model/best` (PhoBERT-base, Test F1 = 0.6711)

---

## 0. OOV Stress-test: Recall In-vocab vs Out-of-Vocabulary

So sánh khả năng nhận diện **kỹ năng đã thấy ở train** vs **kỹ năng hoàn toàn mới** (model phải dựa vào ngữ cảnh):

| Nhóm | Số entity (Test) | Exact Recall | Mức độ lệch |
|---|---|---|---|
| **In-vocab** (đã xuất hiện ở train) | 52/74 (70.3%) | **71.2%** (37/52) | baseline |
| **OOV** (chưa từng thấy ở train) | 22/74 (29.7%) | **54.5%** (12/22) | **−16.6%** |

- Tỉ lệ OOV 29.7% tại Test khớp với dự đoán từ EDA (31.8% skill ở test chưa từng thấy ở train).
- Model vẫn đạt **54.5% recall** trên kỹ năng chưa từng thấy nhờ ngữ cảnh (từ khóa động từ như "sử dụng", "thành thạo", "kinh nghiệm với") — nhưng tụt **16.6 điểm** so với kỹ năng quen thuộc.
- **Kết luận cho pipeline thực tế:** danh sách skill phổ biến nên được build động (từ điển + feedback), vì kỹ năng mới xuất hiện liên tục trong JD. Đây là điểm để thêm "few-shot / dictionary boost" ở phiên bản sau.

> Chạy lại: `python scripts/error_analysis.py`

---

## 1. Bảng phân loại tổng thể các loại lỗi

| Loại lỗi (Error Category) | Số lượng | Tỉ lệ / Tổng lỗi | Bản chất nguyên nhân |
|---|---|---|---|
| **False Negatives (Bỏ sót)** | **10** | 23.8% | Bỏ sót kỹ năng thật (trong đó **5/10 (50.0%)** là do kỹ năng mới OOV) |
| **False Positives (Báo động giả)** | **17** | 40.5% | Mô hình nhận diện nhầm thuật ngữ công việc hoặc danh từ chung thành skill |
| **Boundary Mismatch (Sai ranh giới)** | **15** | 35.7% | Bắt đúng từ khóa chính nhưng cắt thiếu/thừa từ phụ (ví dụ compound skill) |
| **TỔNG CỘNG LỖI** | **42** | 100% | |

---

## 2. Chi tiết từng dạng lỗi & Phân tích nguyên nhân

### A. False Negatives (Bỏ sót thực thể)
Các kỹ năng xuất hiện trong JD nhưng mô hình không nhận diện được:
- **Lỗi do Out-of-Vocabulary (5 trường hợp):**
  - Các kỹ năng: `Function Calling, JPA, Microservice Architecture, Reinforcement Learning Agents`
  - *Nguyên nhân:* Các công cụ mới hoặc ít phổ biến, không xuất hiện trong tập Train (như dự đoán ở bước EDA tỉ lệ OOV = 31.8%). Mô hình không nhận đủ tín hiệu ngữ cảnh để tự tin kích hoạt nhãn `B-SKILL`.
- **Lỗi bỏ sót từ quen thuộc (5 trường hợp):**
  - Các kỹ năng: `Agents, Linux, PostgreSQL, React Native, Vector Database`
  - *Nguyên nhân:* Câu văn có cấu trúc đặc biệt hoặc từ ngữ đứng ở vị trí ngữ pháp không điển hình.

### B. False Positives (Báo động giả / Đoán thừa)
Các từ mô hình gán nhãn là kỹ năng nhưng Ground Truth là nhãn `O`:
- Danh sách từ bị đoán nhầm: `++, /, Adjust, Apple, Blueprints, Boot, CD, Frontend, Google, MMPs, Micro, OOP, Qonversion, SOLID, SYS_OPENAT, SYS_READ, Search`
- *Nguyên nhân:* Thường là các thuật ngữ kỹ thuật chung (như danh từ công nghệ, vị trí công việc, hoặc phương pháp làm việc) khiến mô hình nhầm lẫn là một skill cụ thể.

### C. Boundary Mismatches (Sai ranh giới thực thể)
Mô hình bắt trúng vùng kỹ năng nhưng span bị lệch so với nhãn chuẩn:
- **Thực tế:** `RESTful API`  $\leftrightarrow$  **Mô hình đoán:** `RESTful`
- **Thực tế:** `RESTful API`  $\leftrightarrow$  **Mô hình đoán:** `RESTful`
- **Thực tế:** `Spring Boot`  $\leftrightarrow$  **Mô hình đoán:** `Spring`
- **Thực tế:** `REST API`  $\leftrightarrow$  **Mô hình đoán:** `REST`
- **Thực tế:** `Google Ads`  $\leftrightarrow$  **Mô hình đoán:** `Ad`
- **Thực tế:** `Apple Search Ads`  $\leftrightarrow$  **Mô hình đoán:** `Apple`
- **Thực tế:** `SQL Server`  $\leftrightarrow$  **Mô hình đoán:** `SQL`
- **Thực tế:** `LLM Agents`  $\leftrightarrow$  **Mô hình đoán:** `LLM`
- **Thực tế:** `CI / CD`  $\leftrightarrow$  **Mô hình đoán:** `CI`
- **Thực tế:** `. NET`  $\leftrightarrow$  **Mô hình đoán:** `NET`

- *Nguyên nhân:* Do các kỹ năng dạng ghép nhiều từ (multi-word skills như `.NET Core`, `C/C++`, `Multi - agent`). Mô hình đôi khi chỉ bắt được từ hạt nhân (Head word) hoặc đoán nhãn `B-` cho cả 2 từ liên tiếp thay vì `B-` đi với `I-`.

---

## 3. Các trường hợp lỗi cụ thể (Case Studies)

Dưới đây là một số câu tiêu biểu trong tập Test thể hiện rõ các dạng lỗi trên:

#### Câu 7:
> *"Có kinh nghiệm về kiến ​​ trúc Micro Frontend là một lợi thế lớn ."*

- **Ground Truth:** `[]`
- **Model Pred:**   `['Micro', 'Frontend']`

#### Câu 8:
> *"Implemented system metric parsers (/ proc / stat , / proc / meminfo ) using direct non - blocking Linux system calls ( SYS_OPENAT , SYS_READ ) on fixed stack buffers ."*

- **Ground Truth:** `['Linux']`
- **Model Pred:**   `['SYS_OPENAT', 'SYS_READ']`

#### Câu 10:
> *"Establish engineering best practices , data modeling standards , and comprehensive documentation ( Data Dictionaries , Architecture Blueprints )."*

- **Ground Truth:** `[]`
- **Model Pred:**   `['Blueprints']`

#### Câu 13:
> *"Tham gia phân tích , thiết kế , phát triển và bảo trì các ứng dụng web Full - stack sử dụng Java Spring Boot và Angular ( v12 +)."*

- **Ground Truth:** `['Java', 'Spring Boot', 'Angular']`
- **Model Pred:**   `['Java', 'Spring', 'Boot', 'Angular']`

#### Câu 21:
> *"Tối thiểu 1 , 5 năm React Native ."*

- **Ground Truth:** `['React Native']`
- **Model Pred:**   `[]`

#### Câu 29:
> *"Build and operate end - to - end ELT / ETL data pipelines ( batch & near - realtime ) to ingest data from MMPs ( Adjust , Qonversion ), Ad Networks ( Google Ads , AdMob , AppLovin , Mintegral , Apple Search Ads ,...), and in - app event tracking into the Data Warehouse ."*

- **Ground Truth:** `['ELT', 'ETL', 'Google Ads', 'AdMob', 'AppLovin', 'Mintegral', 'Apple Search Ads']`
- **Model Pred:**   `['ELT', 'ETL', 'MMPs', 'Adjust', 'Qonversion', 'Ad', 'Google', 'AdMob', 'AppLovin', 'Mintegral', 'Apple', 'Search']`

#### Câu 34:
> *"6 . Thực hiện đóng gói , phát hành và cập nhật ứng dụng trên Google Play Store và Apple App Store"*

- **Ground Truth:** `[]`
- **Model Pred:**   `['Apple']`

#### Câu 35:
> *"Có kinh nghiệm thiết kế hệ thống Microservice Architecture ."*

- **Ground Truth:** `['Microservice Architecture']`
- **Model Pred:**   `[]`

---

## 4. Đề xuất giải pháp khắc phục (Actionable Improvements cho v2)

1. **Khắc phục lỗi OOV (False Negative):**
   - Áp dụng kỹ thuật Data Augmentation: Thay thế các skill trong tập Train bằng danh sách từ điển kỹ năng (`docs/skill_vocab.md`) để mô hình học cách nhận biết skill trong nhiều ngữ cảnh đa dạng.
2. **Khắc phục lỗi Boundary Mismatch:**
   - ✅ **ĐÃ LÀM (v1 predict.py):** Cải tiến Regex tiền xử lý: không tách các ký tự dính liền như `C/C++`, `CI/CD`, `C#` (thêm `/`, `+`, `.`, `#`, `_` vào cụm glue).
   - ✅ **ĐÃ LÀM (v1 predict.py):** Gom B-SKILL liền kề thành skill nhiều từ (`Spring` + `Boot` → `Spring Boot`).
   - ✅ **ĐÃ LÀM (v1 predict.py):** Bỏ qua dấu `(`, `)` chen giữa span.
   - Kết hợp tầng hậu xử lý Post-processing thành nguồn document chính.
3. **Khắc phục lỗi False Positive:**
   - Bổ sung các câu văn chứa "từ gây nhiễu" (Negative samples) vào tập Train để phạt mô hình khi đoán bừa các danh từ chung.

---

## 5. Kiểm chứng End-to-end sau fix (predict.py)

Chạy `python src/predict.py` với câu JD mẫu (chứa đầy đủ các dạng skill nhiều từ khó):

> *"Thành thạo ít nhất 1 ngôn ngữ Backend: Ưu tiên Python, Go, C/C++ hoặc Java (Spring Boot). Có kinh nghiệm làm việc với hệ quản trị cơ sở dữ liệu PostgreSQL, MySQL, Redis và triển khai Docker, Kubernetes, CI/CD."*

**Kết quả trước fix (14 skills, 6 lỗi):**

```
1. Python      2. Go        3. C     4. ++     5. Java
6. Spring      7. Boot      8. PostgreSQL  9. MySQL
10. Redis      11. Docker   12. Kubernetes  13. CI   14. CD
```

**Kết quả sau fix (11 skills, 0 lỗi):**

```
1. Python     2. Go        3. C/C++    4. Java      5. Spring Boot
6. PostgreSQL 7. MySQL     8. Redis    9. Docker    10. Kubernetes  11. CI/CD
```

| # | Fix | Lỗi gốc → Kết quả |
|---|---|---|
| 1 | Regex glue `/`, `+` vào từ | `C`+`++` → `C/C++` |
| 2 | Regex glue `/` vào từ | `CI`+`CD` → `CI/CD` |
| 3 | Gom B-SKILL liền kề + bỏ `(` `)` | `Spring`+`Boot` → `Spring Boot` (vẫn giữ `Java` riêng vì có `(` chen giữa) |
| 4 | Strip `.` cuối câu (giữ `.NET`) | `CI/CD.` → `CI/CD` |

**Bài học rút ra:** 3/4 lỗi bắt nguồn từ **tầng tokenization mất mát ký tự** chứ không phải từ chính model NER — điều này cho thấy post-processing và rule đóng vai trò quan trọng không kém model trong pipeline thực tế.

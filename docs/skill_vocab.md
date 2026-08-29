# Skill Vocabulary & Labeling Guidelines

Tài liệu tham chiếu khi label BIO. Khi phân vân, tra file này TRƯỚC.
Vẫn không rõ → ghi `sentence_id` vào mục "Review list" ở cuối file, làm câu khác.

## 1. Labeling rules (binding)

| # | Trường hợp | Quyết định | Lý do |
|---|------------|-----------|-------|
| 1 | Tên vị trí: Backend, Frontend, DevOps, BA, DA, DE, DS, AI Engineer, Fullstack, Manager, Intern... | `O` | Là chức danh, không phải năng lực. JD thường viết "Backend Developer" — "Developer" cũng `O` |
| 2 | Soft skill: teamwork, chủ động, tư duy logic, trách nhiệm, giao tiếp, ownership... | `O` | JobMatch xử lý bằng Gemini o tầng prompt, không qua NER |
| 3 | Skill kèm số version: `Python 3`, `Java 8+`, `Spring Boot 3` | Chỉ label tên: `Python`, `Java`, `Spring Boot`; số = `O` | Gom nhóm đẹp hơn khi normalize |
| 4 | Danh từ chung chung: API, database, framework, testing, monitoring, optimization | `O` | không xác định đủ thành một công nghệ |
| 5 | Công nghệ/tên riêng cụ thể (ngôn ngữ, framework, tool, giao thức, methodology có tên riêng) | `SKILL` | Phần core của bài toán |

**Nguyên tắc phân biệt rule 4 vs 5:** tên riêng/có sẵn → SKILL (`Docker`, `REST`, `Kafka`); danh từ nghề nghiệp chung → O (`caching`, `deployment`, `API`).

## 2. Borderline decisions (đã chốt)

| Thuật ngữ | Quyết định | Lý do |
|-----------|-----------|-------|
| `caching` | `O` | Danh từ hoạt động chung, không phải công nghệ |
| `Production Bug` | `O` | Là đối tượng xử lý, không phải năng lực |
| `GitFlow` | SKILL | Methodology có tên riêng |
| `RAG pipelines` | Chỉ label `RAG`, `pipelines` = `O` | Canonical form là RAG; "pipelines" là danh từ chung |
| `REST API` / `gRPC APIs` | Chi label `REST` / `gRPC`; `API(s)` = `O` | API là danh từ chung (rule 4); giao thức mới là skill. Giữ granular nhỏ để normalize sau này dễ |
| `AI/LLM API` | `AI` = SKILL, `LLM` = SKILL, `API` = `O` | Cũng lý do trên |
| `Message Broker`, `Vector Database` | SKILL | Tên vùng công nghệ chuẩn (khác với "database" thuần danh từ) |
| `AI Safety`, `Model Monitoring`, `Guardrails` | SKILL | Lĩnh vực kỹ thuật có tên cụ thể |
| `C++` | Token `C` = B-SKILL, token `++` = I-SKILL | Tokenizer tách "+"; ghép lại vẫn ra "C ++" — chấp nhận, không sửa |
| Từ có gạch nối (`Multi-agent`) | Label vượt qua token `-` bằng I-SKILL | Tokenizer tách "-" thành token riêng |

## 3. Vocabulary (canonical spelling)

Viết đúng form này khi label. Chữ hoa/thường không quan trọng khi label, nhưng đừng dùng tên.

### Languages
Python, Java, Go, C, C++, C#, JavaScript, TypeScript, SQL, PHP, Kotlin, Swift, Rust, Bash

### Frameworks / Libraries
Spring Boot, Django, Flask, FastAPI, React, Vue, Angular, Node.js, Next.js, .NET, Laravel, Flutter, React Native, JPA, Hibernate, LangChain, LangGraph, Autogen, ADK

### Databases / Storage
PostgreSQL, MySQL, MongoDB, Redis, Elasticsearch, Milvus, pgvector, Snowflake, BigQuery

### Data / ML
Pandas, NumPy, PyTorch, TensorFlow, scikit-learn, Apache Airflow, dbt, Power BI, Tableau, Excel, Spark, Kafka, NLP, Computer Vision, Speech, Video Understanding, RAG, Function Calling, CNN, RNN, Transformer, Attention, GNN, Reinforcement Learning

### DevOps / Cloud / Tools
Docker, Kubernetes, Git, GitFlow, CI/CD, Jenkins, AWS, GCP, Azure, Terraform, Linux, Message Broker, REST, RESTful, gRPC, GraphQL, Microservices

### AI / LLM ecosystem
LLM, GPT, Gemini, Claude, AI Agents, Multi-agent systems, AI Safety, Guardrails, Model Monitoring, Prompt Engineering, Vector Database

### Không label (tham khảo để tránh nhầm)
Backend, Frontend, Fullstack, DevOps (khi là vị trí), API, database, caching, testing, Code Review, Unit Test, Integration Test, Production Bug, teamwork, chủ động, trách nhiệm, tư duy logic, ownership, tiếng Anh, GPA, bằng cấp, giải thưởng

> Luu y: "DevOps" là vị trí → O, nhưng Docker/Kubernetes/Jenkins là skill. "Unit Test" là hoạt động → O, nhưng JUnit/pytest (công cụ) là skill.

## 4. Review list (câu phân vân, xử lý cuối tuần)

| sentence_id | Vấn đề | Đã xử lý? |
|-------------|--------|-----------|
| - | - | - |

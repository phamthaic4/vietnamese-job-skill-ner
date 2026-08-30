# Skill Vocabulary & Labeling Guidelines

Tai lieu tham chieu khi label BIO. Khi phan van, tra file nay TRUOC.
Van khong ro → ghi `sentence_id` vao muc "Review list" o cuoi file, lam cau khac.

## 1. Labeling rules (binding)

| # | Truong hop | Quyet dinh | Ly do |
|---|------------|-----------|-------|
| 1 | Ten vi tri: Backend, Frontend, DevOps, BA, DA, DE, DS, QA, QC, Tester, AI Engineer, Fullstack, Manager, Intern... | `O` | La chuc danh, khong phai nang luc |
| 2 | Soft skill: teamwork, chu dong, tu duy logic, trach nhiem, giao tiep, ownership, ham hoc hoi... | `O` | JobMatch xu ly bang Gemini o tang prompt |
| 3 | Skill kem so version: `Python 3`, `Java 8+`, `Angular 8+`, `.NET Core` | Chi label ten: `Python`, `Java`, `Angular`; so = `O` | Gom nhom dep hon khi normalize |
| 4 | Danh tu chung chung: API, database, framework, testing, monitoring, optimization, UI/UX | `O` | Khong xac dinh du thanh mot cong nghe |
| 5 | Cong nghe/ten rieng cu the (ngon ngu, framework, tool, giao thuc, methodology co ten rieng) | `SKILL` | Phan core cua bai toan |
| 6 | Certification/chung chi: ISTQB, CKA, RHCSA, IELTS, JLPT, TOEIC | `O` | Bang cap/chung chi thuoc tang Gemini (da chot o phan scope) |

**Nguyen tac phan biet rule 4 vs 5:** ten rieng/co san → SKILL (`Docker`, `REST`, `Kafka`); danh tu nghe nghiep chung → O (`caching`, `deployment`, `API`).

## 2. Borderline decisions (da chot)

| Thuat ngu | Quyet dinh | Ly do |
|-----------|-----------|-------|
| `caching` | `O` | Danh tu hoat dong chung |
| `Production Bug` | `O` | La doi tuong xu ly, khong phai nang luc |
| `GitFlow` | SKILL | Methodology co ten rieng |
| `RAG pipelines` | Chi label `RAG`, `pipelines` = `O` | Canonical form la RAG |
| `REST API` / `gRPC APIs` / `LLM API` | Chi label `REST` / `gRPC` / `LLM`; `API(s)` = `O` | API la danh tu chung |
| `Message Broker`, `Vector Database` | SKILL | Ten vung cong nghe chuan |
| `AI Safety`, `Model Monitoring`, `Guardrails` | SKILL | Linh vuc ky thuat co ten cu the |
| `C++` | Token `C` = B-SKILL, token `++` = I-SKILL | Tokenizer tach "+"; chap nhan |
| Tu co gach noi (`Multi-agent`) | Label vuot qua token `-` bang I-SKILL | Tokenizer tach "-" |
| `Agile`, `Scrum`, `SDLC`, `STLC` | `O` | Quy trinh lam viec, khong phai cong nghe (khac GitFlow la ky thuat cu the) |
| `OOP`, `SOLID`, `Design Pattern(s)`, `Clean Architecture` | `OOP`/`SOLID`/`Design Patterns` = `O`; `Clean Architecture` = SKILL | OOP/SOLID la kien thuc nen; Clean Architecture la kien truc co ten rieng |
| `SSR`, `SSG`, `GitOps` | SKILL | Ky thuat co ten rieng |
| `JSON`, `XML`, `HTTP/HTTPS` | `O` | Format/giao thuc nen tang, qua pho thong de co gia tri phan biet |
| `JWT`, `OAuth2`, `TLS/SSL`, `TCP/IP`, `DNS` | SKILL | Tieu chuan bao mat/mang cu the |
| `Test Case`, `Functional Testing`, `Regression Testing`, `Unit Test`, `Bug Reporting` | `O` | Hoat dong kiem thu (rule 4); cong cu thi la SKILL (Selenium, Appium...) |
| `App Store`, `Google Play`, `Jira`, `Redmine`, `Backlog`, `Confluence` | `O` | Kenh phan phoi / cong cu quan ly (Jira vi tri ranh — neu phan van thi O) |
| `RPA`, `AI Coding Assistant`, `Agile transformation` | `O` | Ten linh vuc/hoat dong; cong cu cu the moi la SKILL (Cursor, Copilot, ChatGPT = SKILL) |
| `CI/CD` | SKILL | Da chot tu ban dau, giu nguyen |
| `ChatGPT`, `Cursor`, `GitHub Copilot`, `Perplexity`, `NotebookLM` | SKILL | Cong cu AI cu the co ten |
| `MCP (Model Context Protocol)` | SKILL | Giao thuc co ten rieng |
| `ISO 20022`, `SWIFT`, `NAPAS`, `CITAD`, `AES`, `RSA` | SKILL | Tieu chuan/he thong co ten trong domain payment |

## 3. Vocabulary (canonical spelling)

Viet dung form nay khi label. Chu hoa/thuong khong quan trong khi label, nhung dung dung ten.

### Languages
Python, Java, Go, C, C++, C#, JavaScript, TypeScript, SQL, PHP, Kotlin, Swift, Rust, Bash, Ruby, Dart, Scala

### Frontend
React, ReactJS, Vue, VueJS, Nuxt, Angular, Next.js, Svelte, HTML5, CSS3, SASS, SCSS, Tailwind, RxJS, NgRx, Redux, Zustand, Recoil, Ant Design, Angular Material, Lighthouse, Chrome DevTools, Web Vitals, Bootstrap, jQuery

### Mobile
React Native, Expo, Flutter, Dart, Kotlin, Swift, SwiftUI, Jetpack Compose, Coroutines, CocoaPods, Firebase, MVVM, MVC, Bloc, Cubit, Provider, Riverpod, GetX, Dio, WebSocket, GraphQL

### Frameworks / Backend
Spring Boot, Django, Flask, FastAPI, Node.js, NestJS, Express, .NET, .NET Core, ASP.NET, ASP.NET Core, MVC, Entity Framework, EF Core, OData, Ocelot, SignalR, AJAX, Laravel, JPA, Hibernate, LangChain, LangGraph, Autogen, ADK, Bun, ElysiaJS, Drizzle ORM, Tokio, Axum

### Databases / Storage
PostgreSQL, MySQL, SQL Server, MongoDB, Redis, Elasticsearch, Oracle, SQLite, MariaDB, Couchbase, Milvus, pgvector, Snowflake, BigQuery, ClickHouse, Redshift

### Data / ML
Pandas, NumPy, PyTorch, TensorFlow, FastAI, scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, SHAP, ONNX, Apache Airflow, Celery, dbt, Apache Spark, PySpark, Kafka, Power BI, Tableau, Looker Studio, Streamlit, Metabase, SSIS, SSAS, MDX, Plotly, Jupyter Notebook, NLP, Computer Vision, Speech, Video Understanding, RAG, Function Calling, CNN, RNN, Transformer, Attention, GNN, Reinforcement Learning, ARIMAX, GARCH, MCP

### DevOps / Cloud / Tools
Docker, Docker Compose, Kubernetes, Amazon EKS, EC2, Lambda, ECS, RDS, Git, GitFlow, GitHub Actions, GitLab CI/CD, Jenkins, ArgoCD, GitOps, Terraform, Ansible, Helm, Packer, SonarQube, Nexus, Artifactory, Harbor, Gerrit, AWS, GCP, Azure, CloudWatch, Datadog, New Relic, Prometheus, Grafana, ELK, OpenSearch, OpenTelemetry, Zabbix, k6, Linux, systemd, Lua, Nginx, Apache, Tomcat, Istio, Linkerd, Vault, Message Broker, REST, RESTful, gRPC, Microservices, RabbitMQ, ActiveMQ, SQS, SNS, Kafka, TCP/IP, DNS, TLS, SSL, JWT, OAuth2, virtualenv/venv

### QA / Testing tools
Selenium, Appium, Playwright, Karate, Serenity BDD, Postman, Swagger, OpenAPI, JMeter, Cypress, GNS3, pfSense, VirtualBox

### AI / LLM ecosystem
LLM, GPT, Gemini, Claude, ChatGPT, Cursor, GitHub Copilot, Perplexity, NotebookLM, AI Agents, Multi-agent systems, AI Safety, Guardrails, Model Monitoring, Prompt Engineering, Vector Database

### Payment / Domain standards
ISO 20022, SWIFT, NAPAS, CITAD, AES, RSA

### Khong label (tham khao de tranh nham)
Backend, Frontend, Fullstack, DevOps (khi la vi tri), SRE (khi la vi tri), API, database, caching, testing, Code Review, Unit Test, Integration Test, Production Bug, teamwork, chu dong, trach nhiem, tu duy logic, ownership, tieng Anh, JSON, XML, HTTP, HTTPS, OOP, SOLID, Design Patterns, Agile, Scrum, SDLC, STLC, RPA, UI/UX, responsive design, cross-browser, push notification, deep link, state management, authentication (khi dung chung), authorization, App Store, Google Play, Jira, Redmine, Backlog, ISTQB, CKA, RHCSA, RHCE, IELTS, JLPT, TOEIC, GPA, bang cap, giai thuong

> Luu y: "DevOps" la vi tri → O, nhung Docker/Kubernetes/Jenkins la skill. "Unit Test" la hoat dong → O, nhung JUnit/pytest la skill. "Authentication" chung chung → O, nhung JWT/OAuth2 la SKILL.

## 4. Review list (cau phan van, xu ly cuoi tuan)

| sentence_id | Van de | Da xu ly? |
|-------------|--------|-----------|
| SSR/SSG neu gap nhieu va label kho | Quyet dinh tam thoi: SKILL | - |
| Star Schema, OLAP Cubes | Concept hay skill? Tam thoi O | - |
| Temenos T24 | Product ngach banking, qua hiem | - |
| Telex/VNI/VIQR, TouchGFX, STM32 | Embedded/IME qua chuyen sau, it gap lai | - |
| LAMDA, DREAM (research framework) | Qua niche, khong co trong vocab | - |
| Cisco, Cisco IOS, OSPF, ACL | Networking — tam thoi: OSPF = SKILL, Cisco = O, ACL = O | - |
| Widget, StatelessWidget, BuildContext (Flutter concepts) | Qua granular → O | - |

# Skill Vocabulary & Labeling Guidelines

Tài liệu tham chiếu khi gán nhãn BIO. Khi phân vân, tra file này TRƯỚC.
Vẫn không rõ → ghi `sentence_id` vào mục "Review list" ở cuối file, làm câu khác.

## 1. Labeling rules (bắt buộc)

| # | Trường hợp | Quyết định | Lý do |
|---|------------|-----------|-------|
| 1 | Tên vị trí: Backend, Frontend, DevOps, BA, DA, DE, DS, QA, QC, Tester, AI Engineer, Fullstack, Manager, Intern... | `O` | Là chức danh, không phải năng lực |
| 2 | Soft skill: teamwork, chủ động, tư duy logic, trách nhiệm, giao tiếp, ownership, ham học hỏi... | `O` | JobMatch xử lý bằng Gemini ở tầng prompt |
| 3 | Skill kèm số version: `Python 3`, `Java 8+`, `Angular 8+`, `.NET Core` | Chỉ label tên: `Python`, `Java`, `Angular`; số = `O` | Gom nhóm đẹp hơn khi normalize |
| 4 | Danh từ chung chung: API, database, framework, testing, monitoring, optimization, UI/UX | `O` | Không xác định đủ thành một công nghệ |
| 5 | Công nghệ/tên riêng cụ thể (ngôn ngữ, framework, tool, giao thức, methodology có tên riêng) | `SKILL` | Phần core của bài toán |
| 6 | Certification/chứng chỉ: ISTQB, CKA, RHCSA, IELTS, JLPT, TOEIC | `O` | Bằng cấp/chứng chỉ thuộc tầng Gemini (đã chốt ở phần scope) |

**Nguyên tắc phân biệt rule 4 vs 5:** tên riêng/có sẵn → SKILL (`Docker`, `REST`, `Kafka`); danh từ nghề nghiệp chung → O (`caching`, `deployment`, `API`).

## 2. Borderline decisions (đã chốt)

| Thuật ngữ | Quyết định | Lý do |
|-----------|-----------|-------|
| `caching` | `O` | Danh từ hoạt động chung |
| `Production Bug` | `O` | Là đối tượng xử lý, không phải năng lực |
| `GitFlow` | SKILL | Methodology có tên riêng |
| `RAG pipelines` | Chỉ label `RAG`, `pipelines` = `O` | Canonical form là RAG |
| `REST API` / `gRPC APIs` / `LLM API` | Chỉ label `REST` / `gRPC` / `LLM`; `API(s)` = `O` | API là danh từ chung |
| `Message Broker`, `Vector Database` | SKILL | Tên vùng công nghệ chuẩn |
| `AI Safety`, `Model Monitoring`, `Guardrails` | SKILL | Lĩnh vực kỹ thuật có tên cụ thể |
| `C++` | Token `C` = B-SKILL, token `++` = I-SKILL | Tokenizer tách "+"; chấp nhận |
| Từ có gạch nối (`Multi-agent`) | Label vượt qua token `-` bằng I-SKILL | Tokenizer tách "-" |
| `Agile`, `Scrum`, `SDLC`, `STLC` | `O` | Quy trình làm việc, không phải công nghệ (khác GitFlow là kỹ thuật cụ thể) |
| `OOP`, `SOLID`, `Design Pattern(s)`, `Clean Architecture` | `OOP`/`SOLID`/`Design Patterns` = `O`; `Clean Architecture` = SKILL | OOP/SOLID là kiến thức nền; Clean Architecture là kiến trúc có tên riêng |
| `SSR`, `SSG`, `GitOps` | SKILL | Kỹ thuật có tên riêng |
| `JSON`, `XML`, `HTTP/HTTPS` | `O` | Format/giao thức nền tảng, quá phổ thông để có giá trị phân biệt |
| `JWT`, `OAuth2`, `TLS/SSL`, `TCP/IP`, `DNS` | SKILL | Tiêu chuẩn bảo mật/mạng cụ thể |
| `Test Case`, `Functional Testing`, `Regression Testing`, `Unit Test`, `Bug Reporting` | `O` | Hoạt động kiểm thử (rule 4); công cụ thì là SKILL (Selenium, Appium...) |
| `App Store`, `Google Play`, `Jira`, `Redmine`, `Backlog`, `Confluence` | `O` | Kênh phân phối / công cụ quản lý (Jira vị trí ranh — nếu phân vân thì O) |
| `RPA`, `AI Coding Assistant`, `Agile transformation` | `O` | Tên lĩnh vực/hoạt động; công cụ cụ thể mới là SKILL (Cursor, Copilot, ChatGPT = SKILL) |
| `CI/CD` | SKILL | Đã chốt từ ban đầu, giữ nguyên |
| `ChatGPT`, `Cursor`, `GitHub Copilot`, `Perplexity`, `NotebookLM` | SKILL | Công cụ AI cụ thể có tên |
| `MCP (Model Context Protocol)` | SKILL | Giao thức có tên riêng |
| `ISO 20022`, `SWIFT`, `NAPAS`, `CITAD`, `AES`, `RSA` | SKILL | Tiêu chuẩn/hệ thống có tên trong domain payment |

## 3. Vocabulary (viết chuẩn)

Viết đúng form này khi label. Chữ hoa/thường không quan trọng khi label, nhưng dùng đúng tên.

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
Docker, Docker Compose, Kubernetes, Amazon EKS, EC2, Lambda, ECS, RDS, Git, GitFlow, GitHub Actions, GitLab CI/CD, Jenkins, ArgoCD, GitOps, Terraform, Ansible, Helm, Packer, SonarQube, Nexus, Artifactory, Harbor, Gerrit, AWS, GCP, Azure, CloudWatch, Datadog, New Relic, Prometheus, Grafana, ELK, OpenSearch, OpenTelemetry, Zabbix, k6, Linux, systemd, Lua, Nginx, Apache, Tomcat, Istio, Linkerd, Vault, Message Broker, REST, RESTful, gRPC, Microservices, RabbitMQ, ActiveMQ, SQS, SNS, TCP/IP, DNS, TLS, SSL, JWT, OAuth2

### QA / Testing tools
Selenium, Appium, Playwright, Karate, Serenity BDD, Postman, Swagger, OpenAPI, JMeter, Cypress, GNS3, pfSense, VirtualBox

### AI / LLM ecosystem
LLM, GPT, Gemini, Claude, ChatGPT, Cursor, GitHub Copilot, Perplexity, NotebookLM, AI Agents, Multi-agent systems, AI Safety, Guardrails, Model Monitoring, Prompt Engineering, Vector Database

### Payment / Domain standards
ISO 20022, SWIFT, NAPAS, CITAD, AES, RSA

### Không label (tham khảo để tránh nhầm)
Backend, Frontend, Fullstack, DevOps (khi là vị trí), SRE (khi là vị trí), API, database, caching, testing, Code Review, Unit Test, Integration Test, Production Bug, teamwork, chủ động, trách nhiệm, tư duy logic, ownership, tiếng Anh, JSON, XML, HTTP, HTTPS, OOP, SOLID, Design Patterns, Agile, Scrum, SDLC, STLC, RPA, UI/UX, responsive design, cross-browser, push notification, deep link, state management, authentication (khi dùng chung), authorization, App Store, Google Play, Jira, Redmine, Backlog, ISTQB, CKA, RHCSA, RHCE, IELTS, JLPT, TOEIC, GPA, bằng cấp, giải thưởng

> Lưu ý: "DevOps" là vị trí → O, nhưng Docker/Kubernetes/Jenkins là skill. "Unit Test" là hoạt động → O, nhưng JUnit/pytest là skill. "Authentication" chung chung → O, nhưng JWT/OAuth2 là SKILL.

## 4. Review list (câu phân vân, xử lý cuối tuần)

| sentence_id / vấn đề | Quyết định tạm thời | Đã xử lý? |
|-------------|--------|-----------|
| SSR/SSG nếu gặp nhiều và label khó | Quyết định tạm thời: SKILL | - |
| Star Schema, OLAP Cubes | Concept hay skill? Tạm thời O | - |
| Temenos T24 | Product ngách banking, quá hiếm | - |
| Telex/VNI/VIQR, TouchGFX, STM32 | Embedded/IME quá chuyên sâu, ít gặp lại | - |
| LAMDA, DREAM (research framework) | Quá niche, không có trong vocab | - |
| Cisco, Cisco IOS, OSPF, ACL | Networking — tạm thời: OSPF = SKILL, Cisco = O, ACL = O | - |
| Widget, StatelessWidget, BuildContext (Flutter concepts) | Quá granular → O | - |

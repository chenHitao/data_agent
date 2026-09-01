# data-agent（掌柜问数系统）项目技术文档

> 一句话概括：这是一个基于 **LangGraph** 构建的 **NL2SQL 数据智能体（ChatBI / 问数系统）**。
> 用户输入自然语言问题（如"统计去年各地区的销售总额"），系统通过**多路召回 + LLM 精选 + SQL 生成 + 校验纠错 + 执行**的完整流水线，返回数仓查询结果，全程通过 **SSE** 流式推送进度。

---

## 目录

1. [项目概述](#1-项目概述)
2. [技术栈总览](#2-技术栈总览)
3. [目录结构详解](#3-目录结构详解)
4. [系统架构](#4-系统架构)
5. [离线流程：元数据知识库构建](#5-离线流程元数据知识库构建)
6. [在线流程：一次问数的完整执行链路](#6-在线流程一次问数的完整执行链路)
7. [核心模块逐文件详解](#7-核心模块逐文件详解)
8. [LangGraph Agent 详解（12 个节点）](#8-langgraph-agent-详解12-个节点)
9. [Prompt 工程设计](#9-prompt-工程设计)
10. [数据模型与存储设计](#10-数据模型与存储设计)
11. [配置体系](#11-配置体系)
12. [运行与使用指南](#12-运行与使用指南)
13. [设计亮点与关键技术点](#13-设计亮点与关键技术点)
14. [注意事项与潜在改进点](#14-注意事项与潜在改进点)

---

## 1. 项目概述

### 1.1 项目定位

本项目是一个 **ChatBI（自然语言问数）** 服务的后端实现。业务场景：用户面对一个 MySQL 数据仓库（星型模型：维度表 + 事实表），希望用自然语言直接提问并获得查询结果，而无需手写 SQL。

系统的核心挑战是：LLM 无法预知数据仓库中有哪些表、哪些字段、字段里有哪些值。本项目通过**构建元数据知识库（Schema 元数据 + 向量索引 + 全文索引）**，在每次问数时**动态召回与问题相关的表结构上下文**，再交给 LLM 生成 SQL，从而解决大 Schema 下 Prompt 装不下、准确率低的问题。

### 1.2 示例数据仓库（conf/meta_config.yaml 中定义）

一个经典的电商星型模型：

- 维度表（dim）：`dim_region`（地区）、`dim_customer`（客户）、`dim_product`（商品）、`dim_date`（日期）
- 事实表（fact）：`fact_order`（订单，含 `order_quantity` 销量、`order_amount` 金额两个度量）
- 业务指标：`GMV`（成交总额 = order_amount 求和）、`AOV`（平均订单金额）

### 1.3 一次典型问数的宏观流程

```
用户提问 ──► 关键词抽取(jieba) ──► 三路并行召回(字段/取值/指标) ──► 合并组装表结构
        ──► LLM 精选表与指标 ──► 补充时间/数据库环境 ──► LLM 生成 SQL
        ──► EXPLAIN 校验 ──► (失败则 LLM 纠错一次) ──► 执行 SQL ──► SSE 返回结果
```

---

## 2. 技术栈总览

### 2.1 语言与工程

| 技术 | 版本要求 | 用途 |
|---|---|---|
| Python | ≥ 3.12（使用 PEP 695 泛型语法 `def load_config[T]`） | 开发语言 |
| uv（uv.lock + pyproject.toml） | - | 依赖管理与虚拟环境（`.venv/`） |

### 2.2 Web 服务层

| 技术 | 版本 | 用途 |
|---|---|---|
| FastAPI | ≥ 0.128.0（`fastapi[standard]`） | HTTP 框架，路由、依赖注入、lifespan 生命周期 |
| uvicorn | 随 fastapi[standard] | ASGI 服务器（`main.py` 中 `uvicorn.run`，0.0.0.0:8000） |
| Starlette StreamingResponse | 随 FastAPI | **SSE（text/event-stream）流式响应**，逐步推送 Agent 执行进度 |
| Pydantic v2 | 随 FastAPI | 请求体校验（`QuerySchema`） |

### 2.3 AI / Agent 层（核心）

| 技术 | 版本 | 用途 |
|---|---|---|
| LangGraph | ≥ 1.0.7 | **Agent 编排框架**：StateGraph、条件边、并行节点（fan-out/fan-in）、`Runtime`/`stream_writer` 自定义流式事件 |
| LangChain | ≥ 1.2.7 | LCEL 链式调用（`prompt \| llm \| parser`）、`init_chat_model`、`JsonOutputParser`/`StrOutputParser` |
| langchain-openai（经 `init_chat_model(model_provider="openai")`） | - | 连接 **OpenAI 兼容 API**（配置中为代理地址 `api.openai-proxy.org`，temperature=0） |
| langchain-huggingface | ≥ 1.2.0 | `HuggingFaceEndpointEmbeddings`，连接自部署的 **TEI（Text Embeddings Inference）** 服务 |
| jieba | ≥ 0.42.1 | 中文分词 + TF-IDF 关键词抽取（`jieba.analyse.extract_tags`，按词性过滤） |
| Embedding 模型 | BAAI/**bge-large-zh-v1.5**（1024 维） | 中文语义向量，部署在 `localhost:8081`（TEI 服务） |

### 2.4 存储层

| 技术 | 版本 | 用途 |
|---|---|---|
| MySQL × 2（meta 元数据库 / dw 数据仓库） | - | 元数据持久化 + 最终查询执行 |
| SQLAlchemy 2.0（async） | ≥ 2.0.46 | ORM（DeclarativeBase/Mapped/mapped_column）、异步连接池、`text()` 裸 SQL 执行 |
| asyncmy | ≥ 0.2.11 | MySQL 异步驱动（连接串 `mysql+asyncmy://...`） |
| Qdrant | qdrant-client ≥ 1.16.2（AsyncQdrantClient） | **向量数据库**：字段召回（data-agent-column）、指标召回（data-agent-metric），COSINE 相似度 |
| Elasticsearch 8（async） | ≥ 8, < 9 | **全文检索**：字段取值召回（data-agent-value 索引，**需安装 IK 中文分词插件**，ik_max_word） |

### 2.5 工程支撑

| 技术 | 用途 |
|---|---|
| loguru | 结构化日志，`patch` 动态注入 request_id，控制台 + 文件双输出（10MB 轮转、保留 7 天） |
| contextvars（ContextVar） | 请求级上下文隔离，每个 HTTP 请求生成 uuid 作为 request_id，全链路日志可追踪 |
| OmegaConf | 类型安全配置：`dataclass` 定义 schema + YAML 合并（`structured → merge → to_object`），启动即校验 |
| PyYAML | 将 State 中的结构化信息序列化为 YAML 文本喂给 LLM（比 JSON 更省 token、对 LLM 更友好） |
| dataclasses | 实体层（entities）、配置层（conf）的不可变数据载体 |

---

## 3. 目录结构详解

```
data-agent/
├── main.py                        # FastAPI 应用入口：注册路由 + request_id 中间件 + uvicorn 启动
├── pyproject.toml                 # 项目依赖声明（uv 管理）
├── uv.lock                        # 依赖锁定文件
│
├── conf/                          # ── 配置文件 ──
│   ├── app_config.yaml            # 运行时配置：日志/双 MySQL/Qdrant/Embedding/ES/LLM
│   └── meta_config.yaml           # 业务元数据声明：表、字段(角色/别名/是否同步取值)、指标定义
│
├── prompts/                       # ── LLM Prompt 模板（7 个，.prompt 纯文本）──
│   ├── extend_keywords_for_column_recall.prompt   # 字段召回的关键词扩展
│   ├── extend_keywords_for_value_recall.prompt    # 取值召回的关键词扩展
│   ├── extend_keywords_for_metric_recall.prompt   # 指标召回的关键词扩展
│   ├── filter_table_info.prompt                   # 表+字段精选
│   ├── filter_metric_info.prompt                  # 指标精选
│   ├── generate_sql.prompt                        # NL → SQL 生成
│   └── correct_sql.prompt                         # SQL 纠错
│
└── app/                           # ── 应用源码 ──
    ├── main 无（入口在根目录 main.py）
    │
    ├── agent/                     # ── LangGraph Agent 层（核心业务编排）──
    │   ├── graph.py               # 图定义：12 节点 + 并行/条件边；__main__ 可独立跑图测试
    │   ├── state.py               # DataAgentState：图的状态 Schema（TypedDict，各节点读写的共享数据）
    │   ├── context.py             # DataAgentContext：图的运行时上下文 Schema（依赖注入的仓储/客户端）
    │   ├── llm.py                 # 全局唯一 LLM 实例（init_chat_model，openai 兼容，temperature=0）
    │   └── nodes/                 # 12 个图节点（每个一个 async 函数）
    │       ├── extract_keywords.py    # ① jieba 关键词抽取
    │       ├── recall_column.py       # ② 字段向量召回（Qdrant）
    │       ├── recall_value.py        # ③ 字段取值全文召回（ES）
    │       ├── recall_metric.py       # ④ 指标向量召回（Qdrant）
    │       ├── merge_retrieved_info.py# ⑤ 合并召回结果 → 组装表结构
    │       ├── filter_table.py        # ⑥ LLM 精选表和字段
    │       ├── filter_metric.py       # ⑦ LLM 精选指标
    │       ├── add_extra_context.py   # ⑧ 补充当前时间 + 数据库环境
    │       ├── generate_sql.py        # ⑨ LLM 生成 SQL
    │       ├── validate_sql.py        # ⑩ EXPLAIN 校验 SQL
    │       ├── correct_sql.py         # ⑪ LLM 纠错 SQL
    │       └── execute_sql.py         # ⑫ 执行 SQL 并推送结果
    │
    ├── api/                       # ── HTTP 接口层 ──
    │   ├── routers/query_router.py    # POST /api/query（SSE 流式）
    │   ├── schemas/query_schema.py    # 请求体模型 {"query": str}
    │   └── dependencies.py            # FastAPI 依赖注入链：session → repository → service
    │
    ├── services/                  # ── 服务层 ──
    │   ├── query_service.py           # 在线问数服务：驱动 LangGraph 图 + SSE 包装
    │   └── meta_knowledge_service.py  # 离线服务：构建元数据知识库
    │
    ├── scripts/
    │   └── build_meta_knowledge.py    # 离线构建脚本入口（CLI：-c 指定 meta 配置）
    │
    ├── repositories/              # ── 仓储层（封装所有外部存储访问）──
    │   ├── mysql/meta/
    │   │   ├── meta_mysql_repository.py   # 元数据库读写（表/字段/指标/主外键查询）
    │   │   └── mappers/                   # Model ↔ Entity 转换器（4 个）
    │   ├── mysql/dw/
    │   │   └── dw_mysql_repository.py     # 数仓操作：查字段类型/取值、DB版本、EXPLAIN、执行 SQL
    │   ├── qdrant/
    │   │   ├── column_qdrant_repository.py# data-agent-column 集合：ensure/upsert/search
    │   │   └── metric_qdrant_repository.py# data-agent-metric 集合：ensure/upsert/search
    │   └── es/
    │       └── value_es_repository.py     # data-agent-value 索引：ensure/index/search
    │
    ├── models/                    # ── ORM 模型（SQLAlchemy，meta 库 4 张表）──
    │   ├── base.py                    # DeclarativeBase
    │   ├── table_info_mysql.py        # table_info 表
    │   ├── column_info_mysql.py       # column_info 表
    │   ├── metric_info_mysql.py       # metric_info 表
    │   └── column_metric_mysql.py     # column_metric 关联表（联合主键）
    │
    ├── entities/                  # ── 领域实体（dataclass，贯穿全流程的纯内存对象）──
    │   ├── table_info.py  / column_info.py  / metric_info.py
    │   ├── column_metric.py           # 字段-指标关联
    │   └── value_info.py              # 字段取值（id/value/column_id）
    │
    ├── clients/                   # ── 客户端管理器（单例，init/close 生命周期管理）──
    │   ├── mysql_client_manager.py    # 通用 MySQL 引擎/会话工厂 → meta、dw 两个实例
    │   ├── qdrant_client_manager.py   # AsyncQdrantClient
    │   ├── es_client_manager.py       # AsyncElasticsearch
    │   └── embedding_client_manager.py# HuggingFaceEndpointEmbeddings（TEI 服务）
    │
    ├── conf/                      # ── 配置加载 ──
    │   ├── app_config.py              # AppConfig dataclass 族 + 模块级加载 → app_config 单例
    │   ├── config_loader.py           # 通用加载函数（OmegaConf structured+merge）
    │   └── meta_config.py             # MetaConfig dataclass（tables/metrics）
    │
    ├── core/                      # ── 基础设施 ──
    │   ├── lifespan.py                # FastAPI lifespan：启动 init 5 个客户端，关闭时释放
    │   ├── log.py                     # loguru 配置：格式、request_id 注入、控制台+文件
    │   └── context.py                 # request_id_ctx_var（ContextVar）
    │
    └── prompt/
        └── prompt_loader.py           # load_prompt(name)：读取 prompts/{name}.prompt
```

---

## 4. 系统架构

### 4.1 分层架构图

```mermaid
flowchart LR
    subgraph Client["客户端"]
        U["用户 / 前端"]
    end

    subgraph API层
        R["query_router<br/>POST /api/query (SSE)"]
        DI["dependencies.py<br/>FastAPI 依赖注入"]
    end

    subgraph 服务层
        QS["QueryService<br/>(在线问数)"]
        MKS["MetaKnowledgeService<br/>(离线知识库构建)"]
    end

    subgraph Agent层["Agent 层 (LangGraph)"]
        G["DataAgent StateGraph<br/>12 节点 · 并行 · 条件边"]
    end

    subgraph 仓储层
        RC["ColumnQdrantRepository"]
        RM["MetricQdrantRepository"]
        RE["ValueESRepository"]
        MM["MetaMySQLRepository"]
        MD["DWMySQLRepository"]
    end

    subgraph 基础设施
        MQ1[("MySQL meta<br/>元数据库")]
        MQ2[("MySQL dw<br/>数据仓库")]
        QD[("Qdrant<br/>向量库")]
        ES[("Elasticsearch<br/>全文索引")]
        EMB["TEI Embedding 服务<br/>bge-large-zh-v1.5 :8081"]
        LLM["OpenAI 兼容 LLM API"]
    end

    U -->|"自然语言问题"| R
    R --> DI --> QS --> G
    G --> RC & RM & RE & MM & MD
    RC & RM --> QD
    RE --> ES
    MM --> MQ1
    MD --> MQ2
    G -.->|"LCEL 链调用"| LLM
    G -.->|"关键词向量化"| EMB
    MKS --> MM & MD & RC & RM & RE
```

### 4.2 两种运行形态

| 形态 | 入口 | 用途 |
|---|---|---|
| **离线构建** | `app/scripts/build_meta_knowledge.py` | 一次性（或定期）把 `meta_config.yaml` 声明的表/字段/指标灌入 meta MySQL、Qdrant、ES，形成"元数据知识库" |
| **在线服务** | `main.py`（FastAPI） | 常驻 HTTP 服务，接收问数请求，驱动 LangGraph 图完成检索-生成-执行 |

两条链路**共用同一套仓储层与客户端管理器**，保证读写索引的一致性。

---

## 5. 离线流程：元数据知识库构建

### 5.1 启动命令

```bash
uv run python -m app.scripts.build_meta_knowledge -c conf/meta_config.yaml
```

### 5.2 执行流程

```
build_meta_knowledge.build()                          # app/scripts/build_meta_knowledge.py:20
 ├─ init 5 个客户端（meta MySQL / dw MySQL / Qdrant / Embedding / ES）
 ├─ 创建 5 个仓储 + MetaKnowledgeService
 └─ MetaKnowledgeService.build(conf/meta_config.yaml) # app/services/meta_knowledge_service.py:224
     ├─ ① OmegaConf 加载 YAML → MetaConfig（dataclass 校验）
     ├─ ② 表信息处理（tables 非空时）：
     │    ├─ _save_tables_to_meta_db()
     │    │    ├─ 每张表 → TableInfo(id = 表名)
     │    │    ├─ dw 库执行 SHOW COLUMNS FROM 表 → 获取每个字段真实类型
     │    │    ├─ dw 库 SELECT DISTINCT 字段 LIMIT 10 → 作为字段示例值 examples
     │    │    └─ ColumnInfo(id = "表名.字段名") → 事务写入 meta 库 table_info + column_info
     │    ├─ _save_column_info_to_qdrant()
     │    │    ├─ 确保 Qdrant 集合 data-agent-column（1024 维，COSINE）
     │    │    ├─ 每个字段生成多条向量点：字段名 1 条 + 描述 1 条 + 每个别名各 1 条
     │    │    ├─ 文本批量向量化（batch=10，aembed_documents）
     │    │    └─ 分批 upsert（batch=20），payload = 完整 ColumnInfo JSON
     │    └─ _save_value_info_to_es()
   	     │    ├─ 确保 ES 索引 data-agent-value（value 字段 ik_max_word 分词）
     │    ├─ 仅处理 meta_config 中 sync=true 的字段（低基数维度列，如省份/大区/品类/品牌/性别/会员等级）
     │    ├─ dw 库 SELECT DISTINCT 最多 100,000 个取值
     │    └─ ValueInfo(id = "表.字段.值") 分批 bulk 写入 ES（batch=20）
     └─ ③ 指标信息处理（metrics 非空时）：
          ├─ _save_metrics_to_meta_db()
          │    ├─ MetricInfo(id = 指标名) + ColumnMetric(字段↔指标 关联)
          │    └─ 事务写入 meta 库 metric_info + column_metric
          └─ _save_metric_info_to_qdrant()
               ├─ 确保 Qdrant 集合 data-agent-metric
               └─ 指标名 / 描述 / 别名 分别向量化并 upsert（payload = MetricInfo JSON）
```

### 5.3 流程图

```mermaid
flowchart TD
    S["build_meta_knowledge.py<br/>-c conf/meta_config.yaml"] --> A["初始化 5 个客户端与仓储"]
    A --> B["OmegaConf 加载 meta_config.yaml → MetaConfig"]
    B --> C{"tables 非空?"}
    C -->|是| D1["表/字段信息 → meta MySQL<br/>(含 dw 真实字段类型 + 10 条示例值)"]
    D1 --> D2["字段向量索引 → Qdrant data-agent-column<br/>(名称/描述/每个别名各建一条向量)"]
    D2 --> D3["字段取值 → ES data-agent-value<br/>(sync=true 的列, distinct 值, ik 分词)"]
    B --> E{"metrics 非空?"}
    E -->|是| E1["指标信息 → meta MySQL<br/>(metric_info + column_metric)"]
    E1 --> E2["指标向量索引 → Qdrant data-agent-metric"]
    E2 --> Z["元数据知识库构建完成"]
    D3 --> Z
```

### 5.4 设计意图

- **一个字段多条向量**（名称/描述/别名分别建点）：用户口语（"销售额""收入""大区"）无论命中哪个别名都能召回该字段，大幅提升召回率。
- **示例值（examples）入元数据**：让 LLM 生成 SQL 时能"看到"字段里长什么样，减少幻觉。
- **取值入 ES（sync 开关控制）**：只有低基数的维度列才同步取值（如"华东""华南"），高基数列（订单号、金额）不进索引，兼顾效果与成本。
- **指标单独建库**：GMV/AOV 这类业务口径知识与物理 Schema 解耦，指标召回后可回链到 `relevant_columns`（如 `fact_order.order_amount`）。

---

## 6. 在线流程：一次问数的完整执行链路

### 6.1 HTTP 层

1. **请求**：`POST /api/query`，JSON 体 `{"query": "统计去年各地区的销售总额"}`（`QuerySchema` 校验）。
2. **中间件**（`main.py:17`）：为每个请求生成 uuid 写入 `request_id_ctx_var`，loguru 的 patch 钩子把 request_id 注入到后续所有日志行 —— 全链路日志可按请求追踪。
3. **依赖注入**（`app/api/dependencies.py`）：`Depends` 链组装 `QueryService`：
   - `get_meta_session / get_dw_session`：async with 生成两个 MySQL 会话（请求结束自动归还连接池）；
   - 各仓储 / embedding 客户端注入；
   - `get_query_service` 汇总为 `QueryService` 实例。
4. **响应**：`StreamingResponse(query_service.query(query), media_type="text/event-stream")` —— **SSE 长连接**。

### 6.2 QueryService（app/services/query_service.py:30）

```python
async def query(self, query: str):
    context = DataAgentContext(...)        # 6 个仓储/客户端 → LangGraph context_schema
    state = DataAgentState(query=query)    # 初始 State：只有用户问题
    async for chunk in graph.astream(input=state, context=context, stream_mode="custom"):
        yield f"data: {json.dumps(chunk, ...)}\n\n"   # 每个事件 → 一条 SSE
    # 异常时 yield {"type": "error", "message": ...}
```

关键点：`stream_mode="custom"` 表示**只接收节点内通过 `runtime.stream_writer(...)` 主动发出的事件**（进度/结果），不输出 LLM token 流，前端拿到的是干净的步骤级进度。

### 6.3 LangGraph 图执行（核心）

#### 图拓扑（app/agent/graph.py:47-66）

```mermaid
flowchart TD
    START((START)) --> K["① extract_keywords<br/>jieba 关键词抽取"]
    K -->|"并行 fan-out"| C["② recall_column<br/>字段向量召回 (Qdrant)"]
    K --> V["③ recall_value<br/>取值全文召回 (ES)"]
    K --> M["④ recall_metric<br/>指标向量召回 (Qdrant)"]
    C --> MG["⑤ merge_retrieved_info<br/>合并召回信息"]
    V --> MG
    M --> MG
    MG -->|"并行"| FT["⑥ filter_table<br/>LLM 精选表+字段"]
    MG --> FM["⑦ filter_metric<br/>LLM 精选指标"]
    FT --> AC["⑧ add_extra_context<br/>当前日期 + DB 环境"]
    FM --> AC
    AC --> GS["⑨ generate_sql<br/>LLM 生成 SQL"]
    GS --> VA{"⑩ validate_sql<br/>EXPLAIN 校验"}
    VA -->|"error == None"| EX["⑫ execute_sql<br/>执行并推送结果"]
    VA -->|"error != None"| CO["⑪ correct_sql<br/>LLM 依据报错纠错"]
    CO --> EX
    EX --> END((END))
```

- ②③④ 三个召回节点位于同一"超步（superstep）"，由 LangGraph **异步并行**执行；
- ⑥⑦ 同理并行；条件边用 `add_conditional_edges` 依据 `state["error"]` 决定走向；
- **纠错只进行一轮**（correct_sql → execute_sql，不再回环校验）。

#### 各节点细节

**① extract_keywords**（`app/agent/nodes/extract_keywords.py`）

- `jieba.analyse.extract_tags(query, allowPOS=...)` 基于 TF-IDF 抽取关键词，并限定词性：名词(n)、人名(nr)、地名(ns)、机构名(nt)、专有名词(nz)、动词(v)、名动词(vn)、形容词(a)、名形词(an)、英文(eng)、成语(i)、习惯用语(l)；
- `keywords = set(jieba关键词 + [完整query])` —— 完整原句也作为一个"关键词"参与召回；
- 写入 State：`keywords`。

**②③④ 三路召回（结构一致）**：每个节点先做 **LLM 查询扩展**，再做**检索**：

| 节点 | 扩展 Prompt | 扩展目标 | 检索引擎 | 检索方式 | 阈值/TopK | 写入 State |
|---|---|---|---|---|---|---|
| recall_column | extend_keywords_for_column_recall | "回答该问题必需的字段概念" | Qdrant data-agent-column | 关键词 → 1024 维向量 → 近邻 | 0.6 / 5 | `retrieved_columns` |
| recall_value | extend_keywords_for_value_recall | "可能存在于字段取值中的候选值" | Elasticsearch data-agent-value | match 全文（ik_max_word） | min_score 0.6 / 5 | `retrieved_values` |
| recall_metric | extend_keywords_for_metric_recall | "指标概念检索词" | Qdrant data-agent-metric | 关键词向量 → 近邻 | 0.6 / 5 | `retrieved_metrics` |

每个节点内部逻辑（以 recall_column 为例，`app/agent/nodes/recall_column.py:31-49`）：

```
chain = PromptTemplate | llm | JsonOutputParser
result = await chain.ainvoke({"query": query})     # LLM 扩展出的关键词 JSON 数组
keywords = set(jieba关键词 + [query] + LLM扩展词)   # 三源合并
for keyword in keywords:                            # 逐词检索
    embedding = await embedding_client.aembed_query(keyword)
    payloads = await column_qdrant_repository.search(embedding)
    # 以 payload.id（"表.字段"）为 key 去重合并
```

**⑤ merge_retrieved_info**（`app/agent/nodes/merge_retrieved_info.py`）—— 把三路散装召回"缝"成完整 Schema 上下文：

1. **指标回链字段**：指标的 `relevant_columns`（如 `fact_order.order_amount`）若未在召回字段中，从 meta 库补查；
2. **取值归位**：每个召回的 `ValueInfo`（value + column_id）找到所属字段，把取值 append 到该字段的 `examples`（让 LLM 看到用户问题里的"华东"确实存在于 region_name 列）；
3. **按表分组**：`column.table_id → [columns]` 得到表-字段映射；
4. **显式补主外键**：对每张表调 `get_key_columns_by_table_id`（role ∈ primary_key/foreign_key）补齐关联键 —— **保证 LLM 知道表之间怎么 JOIN**；
5. 组装 `TableInfoState`（表名/角色/描述/字段列表）与 `MetricInfoState`，写入 State 的 `table_infos` / `metric_infos`。

**⑥⑦ LLM 精选（filter_table / filter_metric）**

- 输入：用户问题 + `yaml.dump(...)` 序列化后的候选信息（YAML 比 JSON 短且可读，利于 LLM）；
- filter_table 输出 `{"fact_order": ["order_amount", "region_id"], "dim_region": [...]}` 形式的 JSON 对象，代码据此对 State 中的表列表、字段列表做**双重裁剪**（未选中的表整体移除，表内未选中的字段移除）；
- filter_metric 输出指标名数组（允许空数组），只保留被选中的指标；
- 意义：召回阶段宁可多勿漏（高召回），此阶段精挑细选（高精度），**把最小必要的 Schema 喂给 SQL 生成**，防止 LLM 分心与幻觉。

**⑧ add_extra_context**：写入两类"额外上下文"：

- `date_info`：当前日期（`%Y-%m-%d`）、英文星期（`%A`）、季度（`Q{n}`）—— 让 LLM 能把"去年""最近三个月"解析成具体时间范围；
- `db_info`：dw 库 `SELECT version()` 的版本号 + SQLAlchemy dialect 名（`mysql`）—— 约束生成 SQL 的语法方言。

**⑨ generate_sql**：LCEL 链 `generate_sql.prompt | llm | StrOutputParser`，输入为 query + 四块 YAML 上下文，输出**纯 SQL 文本**（Prompt 强制禁止 markdown 代码块包裹），写入 State 的 `sql`。

**⑩ validate_sql**：调用 `DWMySQLRepository.validate_sql`，即 `EXPLAIN <sql>` —— **不真正执行，只让 MySQL 解析器验证语法/表字段存在性/JOIN 合法性**：

- 成功 → `return {"error": None}`；
- 失败 → 捕获异常，`return {"error": str(e)}`（注意：不 raise，让错误进入 State 供纠错节点使用）。

**条件边**（`app/agent/graph.py:61-63`）：`"execute_sql" if state["error"] is None else "correct_sql"`。

**⑪ correct_sql**：Prompt 携带完整上下文 + 原 SQL + 数据库报错信息，要求"最小必要修复、语义不变、禁止改动指标口径"，输出新 SQL 覆盖 State 的 `sql`，随后直接进入执行（单轮修复）。

**⑫ execute_sql**：`DWMySQLRepository.execute_sql` 真正执行 SELECT，结果转 `list[dict]`，通过 `stream_writer` 发出**最终结果事件**：`{"type": "result", "data": [...]}`。节点本身不修改 State（图到此结束）。

### 6.4 SSE 事件流示例

```
data: {"type": "progress", "step": "抽取关键字",     "status": "running"}
data: {"type": "progress", "step": "抽取关键字",     "status": "success"}
data: {"type": "progress", "step": "召回字段",       "status": "running"}   # 三路召回并发，事件交错出现
data: {"type": "progress", "step": "召回字段取值",   "status": "running"}
data: {"type": "progress", "step": "召回指标",       "status": "running"}
data: {"type": "progress", "step": "召回指标",       "status": "success"}
...
data: {"type": "progress", "step": "合并召回信息",   "status": "success"}
data: {"type": "progress", "step": "过滤表格",       "status": "running"}
data: {"type": "progress", "step": "过滤指标",       "status": "running"}
...
data: {"type": "progress", "step": "生成SQL",        "status": "success"}
data: {"type": "progress", "step": "验证SQL",        "status": "success"}   # 或 error → 校正SQL
data: {"type": "progress", "step": "执行SQL",        "status": "success"}
data: {"type": "result", "data": [{"region_name": "华东", "total_amount": 123456.0}, ...]}
```

异常路径最终会收到：`data: {"type": "error", "message": "..."}`。

---

## 7. 核心模块逐文件详解

### 7.1 main.py（应用入口）

- `FastAPI(lifespan=lifespan)` 挂载生命周期；
- `app.include_router(query_router)` 注册 `/api/query`；
- `@app.middleware("http")`：每个请求先 `request_id_ctx_var.set(uuid.uuid4())` 再放行 —— 配合 loguru 实现请求级日志追踪；
- `__main__` 时 `uvicorn.run(app, host="0.0.0.0", port=8000)`。

### 7.2 app/core/

| 文件 | 内容 |
|---|---|
| `context.py` | `request_id_ctx_var = ContextVar("request_id", default="1")`，基于标准库 contextvars，异步任务间天然隔离 |
| `log.py` | loguru：`logger.remove()` 清默认 sink → `patch(inject_request_id)` 给每条日志打上 request_id → 控制台 sink + 文件 sink（`logs/app.log`，10MB 轮转、保留 7 天、utf-8），格式含时间/级别/request_id/模块：函数：行号 |
| `lifespan.py` | `@asynccontextmanager`：启动时依序 `init()` 5 个客户端（Embedding/Qdrant/ES/meta-MySQL/dw-MySQL）；应用关闭时依序 `await close()` 释放（连接池 dispose、客户端 close） |

### 7.3 app/conf/（配置体系）

- `app_config.py`：用 dataclass 定义完整配置 Schema（`AppConfig` ← LoggingConfig/DBConfig/QdrantConfig/EmbeddingConfig/ESConfig/LLMConfig），模块导入时执行 `OmegaConf.merge(dataclass schema, yaml)` → `to_object` 得到强类型 `app_config` 单例 —— **YAML 缺字段/类型错会在启动瞬间报错**；
- `config_loader.py`：泛型函数 `load_config[T](file, schema_cls)`（PEP 695 语法）封装同一套路；
- `meta_config.py`：`MetaConfig(tables: list[TableConfig], metrics: list[MetricConfig])`，字段级配置含 `role`（primary_key/foreign_key/dimension/measure）、`alias` 别名数组、`sync` 是否同步取值到 ES。

### 7.4 app/clients/（客户端管理器）

统一模式：**类封装 + 模块级单例 + `init()` 惰性初始化 + `close()` 异步释放**。

| 管理器 | 产物 | 关键配置 |
|---|---|---|
| `MysqlClientManager`（实例 ×2：meta/dw） | `AsyncEngine` + `async_sessionmaker` | URL `mysql+asyncmy://user:pwd@host:port/db?charset=utf8mb4`；`pool_size=10`、`pool_pre_ping=True`（防连接失效）、`autoflush=True`、`expire_on_commit=False`、`autobegin=True` |
| `QdrantClientManager` | `AsyncQdrantClient` | `http://host:6333` |
| `ESClientManager` | `AsyncElasticsearch` | `http://host:9200` |
| `EmbeddingClientManager` | `HuggingFaceEndpointEmbeddings(model="http://host:8081")` | 指向 TEI 服务加载的 bge-large-zh-v1.5 |

每个文件都带 `__main__` 自测代码（建集合/索引、插入、检索的最小示例），可单独运行验证连通性。

### 7.5 app/entities/（领域实体）与 app/models/（ORM 模型）

- entities：纯 `@dataclass`（ColumnInfo/ValueInfo/MetricInfo/TableInfo/ColumnMetric），是 Agent、仓储、Qdrant/ES payload 共用的"通用语言"；
- models：SQLAlchemy `DeclarativeBase` 子类，映射 meta 库 4 张表（见第 10 节）；
- mappers（`app/repositories/mysql/meta/mappers/`）：每对 Entity↔Model 一个静态方法转换器（`to_entity` / `to_model`，借助 `dataclasses.asdict`），隔离 ORM 与业务层。

### 7.6 app/repositories/（仓储层）

**MetaMySQLRepository**（meta 库）：

- `save_table_infos / save_column_infos / save_metric_infos / save_column_metrics`：批量 `session.add_all`（配合事务上下文写入）；
- `get_column_info_by_id / get_table_info_by_id`：`session.get(Model, id)` 主键查；
- `get_key_columns_by_table_id(table_id)`：`where table_id=? and role in ('primary_key','foreign_key')` —— 为合并节点补 JOIN 键。

**DWMySQLRepository**（dw 库，面向数仓）：

| 方法 | SQL | 用途 |
|---|---|---|
| `get_column_types(table)` | `SHOW COLUMNS FROM {table}` | 知识库构建：取真实字段类型 |
| `get_column_values(table, col, limit)` | `SELECT DISTINCT {col} FROM {table} LIMIT {n}` | 构建：示例值(10) / ES 取值(10万)；运行时不使用 |
| `get_db_info()` | `SELECT version()` + dialect 名 | 生成 SQL 的环境上下文 |
| `validate_sql(sql)` | `EXPLAIN {sql}` | 语法/结构预检（不真正执行） |
| `execute_sql(sql)` | `text(sql)` + `mappings()` | 执行查询，返回 `list[dict]` |

**ColumnQdrantRepository / MetricQdrantRepository**（泛型同构）：

- `ensure_collection`：不存在则建（`VectorParams(size=1024, distance=COSINE)`）；
- `upsert(ids, embeddings, payloads, batch=20)`：分批 `PointStruct` 写入；
- `search(embedding, score_threshold=0.6, limit=5)`：`query_points` 余弦近邻，payload 反序列化为 Entity 返回。

**ValueESRepository**：

- `ensure_index`：建 `data-agent-value`，`dynamic: false`，value 字段 `ik_max_word` 建分词 + 检索分词；
- `index(value_infos, batch=20)`：bulk 批量写入（`_id` = ValueInfo.id，天然幂等覆盖）；
- `search(keyword, min_score=0.6, size=5)`：`match` 全文检索，按 BM25 相关性过滤低分结果，返回 `ValueInfo` 列表。

### 7.7 app/api/（接口层）

- `query_schema.py`：`QuerySchema(BaseModel): query: str`；
- `query_router.py`：`POST /api/query` → `StreamingResponse(query_service.query(query.query), media_type="text/event-stream")`；
- `dependencies.py`：完整 DI 链（见 6.1），注意 MySQL 会话用 `async with ... as session: yield session` 保证请求级生命周期。

### 7.8 app/services/

- `QueryService`（在线）：见 6.2；持有 6 个依赖，每次查询构造全新的 `DataAgentContext` 传入图（LangGraph context_schema 注入）；
- `MetaKnowledgeService`（离线）：见第 5 节，构建动作全部包在 `async with session.begin():` 事务中。

---

## 8. LangGraph Agent 详解（12 个节点）

### 8.1 State 与 Context 的职责分离（LangGraph 1.x 设计）

- **`DataAgentState`**（`app/agent/state.py`）：图内流转的"业务数据"，TypedDict 定义，节点返回 dict 与 State 合并（LangGraph 默认覆盖合并策略）：

| 字段 | 类型 | 产生者 | 消费者 |
|---|---|---|---|
| `query` | str | 入口 | 所有 LLM 节点 |
| `keywords` | list[str] | extract_keywords | 三路召回 |
| `retrieved_columns` | list[ColumnInfo] | recall_column | merge_retrieved_info |
| `retrieved_values` | list[ValueInfo] | recall_value | merge_retrieved_info |
| `retrieved_metrics` | list[MetricInfo] | recall_metric | merge_retrieved_info |
| `table_infos` | list[TableInfoState] | merge_retrieved_info / filter_table | filter_table、generate_sql、correct_sql |
| `metric_infos` | list[MetricInfoState] | merge_retrieved_info / filter_metric | filter_metric、generate_sql、correct_sql |
| `date_info` | DateInfoState | add_extra_context | generate_sql、correct_sql |
| `db_info` | DBInfoState | add_extra_context | generate_sql、correct_sql |
| `sql` | str | generate_sql / correct_sql | validate_sql、correct_sql、execute_sql |
| `error` | str \| None | validate_sql | 条件边、correct_sql |

- **`DataAgentContext`**（`app/agent/context.py`）：**运行时依赖**（6 个客户端/仓储），与 State 分离 —— 不会被序列化/持久化，也不参与节点间的数据流，节点通过 `runtime.context[...]` 获取。好处：State 保持纯净可检查/可回放，依赖可替换（换库只需换注入对象）。

### 8.2 节点一览表

| # | 节点 | 是否 LLM | 读写 | 外部系统 | 使用的 Prompt |
|---|---|---|---|---|---|
| 1 | extract_keywords | 否（jieba） | 写 keywords | - | - |
| 2 | recall_column | 是（扩展词） | 写 retrieved_columns | LLM + Embedding + Qdrant | extend_keywords_for_column_recall |
| 3 | recall_value | 是（扩展词） | 写 retrieved_values | LLM + ES | extend_keywords_for_value_recall |
| 4 | recall_metric | 是（扩展词） | 写 retrieved_metrics | LLM + Embedding + Qdrant | extend_keywords_for_metric_recall |
| 5 | merge_retrieved_info | 否 | 写 table_infos、metric_infos | meta MySQL | - |
| 6 | filter_table | 是 | 改 table_infos | LLM | filter_table_info |
| 7 | filter_metric | 是 | 改 metric_infos | LLM | filter_metric_info |
| 8 | add_extra_context | 否 | 写 date_info、db_info | dw MySQL（version） | - |
| 9 | generate_sql | 是 | 写 sql | LLM | generate_sql |
| 10 | validate_sql | 否 | 写 error | dw MySQL（EXPLAIN） | - |
| 11 | correct_sql | 是 | 写 sql | LLM | correct_sql |
| 12 | execute_sql | 否 | 无（writer 推送 result） | dw MySQL（执行） | - |

### 8.3 全部 12 个节点都会做的一件事：stream_writer 进度上报

每个节点第一行 `writer = runtime.stream_writer`，进入时 `writer({"type":"progress","step":...,"status":"running"})`，结束/异常时发 `success` / `error`。这就是 SSE 进度事件的来源，前端可实时渲染执行进度条。

### 8.4 图的运行入口

- 生产：`QueryService.query` → `graph.astream(input=state, context=context, stream_mode="custom")`；
- 本地调试：直接运行 `python -m app.agent.graph`（`graph.py` 的 `__main__` 块自行 init 客户端、构造 context，测试问题写死为"统计去年各地区的销售总额"，逐 chunk 打印）。

---

## 9. Prompt 工程设计

7 个 Prompt 全部是中文、角色导向（"你是一名XX专家"）、带硬性规则与 few-shot 示例、只要求结构化输出。存放于 `prompts/`，由 `app/prompt/prompt_loader.py` 的 `load_prompt(name)` 按文件名读取。

| Prompt | 角色设定 | 核心规则 | 输出格式 |
|---|---|---|---|
| extend_keywords_for_column_recall | 数据表字段推断专家 | 只输出"回答问题必需的最小字段概念集合"；禁止取值/表名/SQL；时间/人群/对比等语义必须映射到对应字段 | JSON 数组（中文业务字段名） |
| extend_keywords_for_value_recall | 业务语义解析专家 | 只输出"可能存在于字段取值空间"的候选值（枚举/实体/时间语义词/人群标签）；来源于问题显式或隐含语义 | JSON 数组 |
| extend_keywords_for_metric_recall | 指标语义扩展专家 | 只输出"指标概念短语"（如转化率/GMV/客单价）；允许同义词与英文缩写；禁止字段名/表名/取值 | JSON 数组 |
| filter_table_info | 查询规划专家 | 只能从候选中选择，禁止新增/改名；每张选中表至少含一个选中字段；以"是否实际使用"为唯一标准；多表必须包含 JOIN 主外键 | JSON 对象 `{"表名":["字段",...]}`（模板中用 `{{}}` 转义示例） |
| filter_metric_info | 指标筛选专家 | 最小指标集合；无度量诉求必须返回空数组；仅筛选分组用的不算指标 | JSON 数组（可空） |
| generate_sql | 资深数据库专家+数据分析师 | 只准用提供的表字段；有指标定义必须遵循其口径；只读查询；单条 SQL；**输出纯文本 SQL，禁止 ``` 包裹**；语法符合 db_info 声明的方言版本 | 纯 SQL |
| correct_sql | 数据库专家+SQL 调试专家 | 基于报错最小修复；语义/口径/时间范围不得改变；保持结构稳定；同样只读、单条、纯文本 | 纯 SQL |

**Prompt 工程共性亮点**：

1. 三个扩展 Prompt 互相正交（字段概念 / 取值候选 / 指标概念），避免一路召回污染另一路；
2. 反复强调"最小集合""禁止兜底""禁止依赖外部知识"——抑制 LLM 过度发散；
3. 输出全部走 `JsonOutputParser` 或 `StrOutputParser`，与 LCEL 链无缝衔接；
4. SQL 类 Prompt 明令禁止 markdown 代码块，规避 StrOutputParser 拿到 ```sql 围栏的问题。

---

## 10. 数据模型与存储设计

### 10.1 meta MySQL（元数据库，4 张表）

**table_info**（表注册表）

| 列 | 类型 | 说明 |
|---|---|---|
| id | String(64) PK | 表名（构建时 id = name） |
| name | String(128) | 表名 |
| role | String(32) | fact / dim |
| description | Text | 表描述（喂给 LLM） |

**column_info**（字段注册表）

| 列 | 类型 | 说明 |
|---|---|---|
| id | String(64) PK | 全局唯一：`表名.字段名`（如 `fact_order.order_amount`） |
| name | String(128) | 字段名 |
| type | String(64) | 从 dw 库 SHOW COLUMNS 采集的真实类型 |
| role | String(32) | primary_key / foreign_key / dimension / measure |
| examples | JSON | 示例值数组（构建时取 10 条；运行时会被召回的取值继续充实） |
| description | Text | 字段描述 |
| alias | JSON | 别名数组（如 销售额/订单金额/收入） |
| table_id | String(64) | 所属表 id |

**metric_info**（指标注册表）

| 列 | 类型 | 说明 |
|---|---|---|
| id | String(64) PK | 指标名（GMV/AOV） |
| name / description | String(128) / Text | 名称与业务口径描述 |
| relevant_columns | JSON | 关联字段 `["fact_order.order_amount"]` |
| alias | JSON | 指标别名（成交总额/订单总额） |

**column_metric**（字段-指标关联表）：联合主键 `(column_id, metric_id)`。

### 10.2 Qdrant（向量库，2 个 collection）

| Collection | 向量规格 | 一条业务记录 → 多少个向量点 | payload |
|---|---|---|---|
| data-agent-column | 1024 维 / COSINE | 字段名 1 + 描述 1 + 每个别名各 1 | 完整 ColumnInfo |
| data-agent-metric | 1024 维 / COSINE | 指标名 1 + 描述 1 + 每个别名各 1 | 完整 MetricInfo |

> 每个点的 id 是 uuid4（同一字段的多条向量点各自独立），检索命中后靠 payload.id（`表.字段`）去重还原成业务实体。

### 10.3 Elasticsearch（全文索引，1 个 index）

**data-agent-value**（`dynamic: false`）：

| 字段 | 类型 | 说明 |
|---|---|---|
| id | keyword | `表.字段.值` |
| value | text（analyzer/search_analyzer = **ik_max_word**） | 字段取值原文 |
| column_id | keyword | 所属字段 `表.字段` |

> 依赖 ES 安装 **IK 中文分词插件**；写入按 `_id` 幂等。

### 10.4 dw MySQL（数据仓库）

业务库本身，星型模型（5 表），系统对它只做读操作（SHOW COLUMNS / SELECT DISTINCT / SELECT version / EXPLAIN / SELECT）。meta_config.yaml 中 `sync: true` 的维度列（省份、大区、国家、客户名、性别、会员等级、商品名、品类、品牌、季度）其取值会被同步进 ES。

---

## 11. 配置体系

### 11.1 conf/app_config.yaml（运行时配置）

| 配置段 | 字段 | 当前值 | 用途 |
|---|---|---|---|
| logging.file | enable/level/path/rotation/retention | true / INFO / logs / 10 MB / 7 days | 文件日志 |
| logging.console | enable/level | true / INFO | 控制台日志 |
| db_meta | host/port/user/password/database | localhost:3306 / atguigu / meta | 元数据库连接 |
| db_dw | 同上 | localhost:3306 / atguigu / dw | 数仓连接 |
| qdrant | host/port/embedding_size | localhost:6333 / **1024** | 向量库（维度须与 embedding 模型一致） |
| embedding | host/port/model | localhost:8081 / BAAI/bge-large-zh-v1.5 | TEI 向量化服务 |
| es | host/port/index_name | localhost:9200 / data_agent | ES 连接（注：索引名实际由仓储类写死为 data-agent-value，此配置项未参与） |
| llm | model_name/api_key/base_url | gpt-5.2-codex / \<api-key\> / https://api.openai-proxy.org/v1 | OpenAI 兼容 LLM |

> ⚠️ 该文件含数据库明文密码与 API Key 占位，生产环境应改用环境变量/密钥管理，且不要提交真实密钥到仓库。

### 11.2 conf/meta_config.yaml（业务元数据声明）

声明式地描述"系统认识哪些表/字段/指标"：

- 表：name、role（dim/fact）、description、columns[]（name、role、description、alias、sync）；
- 指标：name、description、relevant_columns、alias。

修改业务模型后重跑构建脚本即可重建知识库（注意：当前实现为追加写入，重复执行会产生重复数据，见第 14 节）。

---

## 12. 运行与使用指南

### 12.1 前置依赖

1. **MySQL**（两库）：`meta`（需预先建好 4 张元数据表）与 `dw`（星型模型业务数据）；
2. **Qdrant**：默认 `localhost:6333`（如 `docker run -p 6333:6333 qdrant/qdrant`）；
3. **Elasticsearch 8**：默认 `localhost:9200`，**必须安装 IK 分词插件**（analysis-ik）；
4. **TEI Embedding 服务**：默认 `localhost:8081`，加载 `BAAI/bge-large-zh-v1.5`（例如官方 `ghcr.io/huggingface/text-embeddings-inference` 镜像）；
5. **LLM**：任意 OpenAI 兼容 API，改 `conf/app_config.yaml` 的 `llm` 段。

### 12.2 安装与启动

```bash
# 1. 安装依赖（uv 自动创建 .venv 并按 uv.lock 锁定版本）
uv sync

# 2. 构建元数据知识库（离线，一次性）
uv run python -m app.scripts.build_meta_knowledge -c conf/meta_config.yaml

# 3. 启动在线服务（0.0.0.0:8000）
uv run python main.py
```

### 12.3 调用接口

```bash
curl -N -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "统计去年各地区的销售总额"}'
```

`-N` 禁用缓冲以持续接收 SSE 流。事件序列见 6.4。

### 12.4 其它自测入口

| 命令 | 作用 |
|---|---|
| `uv run python -m app.agent.graph` | 独立跑一遍 Agent 图（绕过 HTTP），打印每个 stream chunk |
| `uv run python -m app.conf.app_config` | 校验配置加载 |
| `uv run python -m app.clients.qdrant_client_manager` 等 | 各客户端连通性自测 |
| `uv run python -m app.core.log` | 验证 request_id 日志注入（并发两请求的模拟） |

---

## 13. 设计亮点与关键技术点

1. **混合检索（Hybrid Retrieval）三路并行**：
   - 关键词来源三合一：jieba TF-IDF（lexical）+ LLM 扩展（semantic query expansion，三个方向正交的 Prompt）+ 原始整句；
   - 检索引擎二合一：Qdrant 向量近邻（字段/指标，语义层）+ ES 全文 BM25（取值，精确词面层），各司其职；
   - 一个字段以"名称/描述/别名"多条向量入索引，命中任一别名即可召回。

2. **两阶段"召回-精选"漏斗**：召回阶段宽进（阈值 0.6/Top5/多词多轮），merge 后由 LLM 做最小集合精选（filter_table/filter_metric），最终喂给 generate_sql 的上下文既完整又精简 —— 这是解决"大 Schema 装不下 Prompt + LLM 选择困难"的经典架构。

3. **SQL 自修复闭环（Generate → Validate → Correct）**：用 `EXPLAIN` 做轻量预检（不执行、无副作用），失败则携带原始报错让 LLM 做受约束的最小修复（Prompt 严禁改变业务口径），修一次直接执行 —— 工程成本与成功率的平衡点。

4. **LangGraph 1.x 特性运用**：
   - `state_schema`（业务数据）与 `context_schema`（依赖注入）分离；
   - 节点签名 `async def node(state, runtime: Runtime[Context])`，`runtime.stream_writer` 实现**步骤级自定义流**（`stream_mode="custom"`），天然适配 SSE；
   - 扇出节点（三路召回、双过滤）在同一超步内自动并发，无需手写 asyncio.gather；
   - `add_conditional_edges` 用 lambda 依据 `state["error"]` 路由。

5. **LLM 友好的上下文序列化**：表结构/指标/时间/DB 环境统一 `yaml.dump(allow_unicode=True, sort_keys=False)` —— 相比 JSON，YAML 无引号噪音、省 token、层级清晰。

6. **可观测性**：ContextVar + loguru.patch 让每个请求的 uuid 出现在**所有**日志行（含异步任务），配合节点的 running/success/error 事件，前后端均可定位卡点。

7. **类型安全配置**：OmegaConf `structured(dataclass) + merge(yaml) + to_object`，配置错误在进程启动时即失败；`load_config[T]` 使用 Python 3.12 PEP 695 泛型函数语法。

8. **分层清晰、依赖倒置**：Agent 节点不直接 import 客户端，全部经 Context 拿仓储；仓储层统一封装外部 I/O；entities（dataclass）与 models（ORM）经 mapper 隔离 —— 每一层都可单测/替换。

9. **SSE 优先的交互协议**：`text/event-stream` + `data: {...}\n\n` 帧格式，前端无需 WebSocket 即可渐进渲染进度与结果；错误也走流内事件（`{"type":"error"}`），连接语义统一。

10. **幂等/批量的写入设计**：ES bulk 以 `_id` 覆盖写；Qdrant upsert 天然幂等；embedding 与向量写入均分批（10/20）防止大请求。

---

## 14. 注意事项与潜在改进点

以下为代码审阅中发现的值得注意的点（不代表缺陷，视生产要求决定是否处理）：

1. **知识库重复构建无清理**：`build()` 只 upsert/append，不清理旧数据。重复执行脚本会向 meta MySQL 插入重复行（主键冲突报错）、向 Qdrant/ES 累积 uuid 不同的重复向量点。建议：构建前 drop/recreate，或对 meta 库改用 upsert。

2. **ES 索引名配置脱节**：`app_config.es.index_name: data_agent` 实际未被使用；`ValueESRepository.index_name` 硬编码为 `data-agent-value`。

3. **`merge_retrieved_info` 的空值风险**：`get_column_info_by_id` 可能返回 `None`（如 metric 的 relevant_columns 指向不存在的列），后续 `retrieved_columns_map[column_id].examples` 会 `AttributeError`。

4. **纠错仅一轮**：correct_sql 后不再验证，若纠错仍失败将直接抛异常终止（SSE 收到 error）。可改为有限次数的 validate↔correct 循环。

5. **`correct_sql` 节点的 `input_variables` 声明不全**：模板实际含 7 个变量，声明只有 `["query", "metric_infos"]`；因运行时 ainvoke 传入了全部变量才未出错，属依赖 LangChain 宽松校验的隐患。

6. **SQL 直通执行的安全边界**：LLM 生成的 SQL 经 `text()` 直接执行。当前防线是 Prompt（只读、单条）；更稳妥可在 validate/execute 前加 AST/正则白名单（仅允许 SELECT、禁多语句、禁注释注入）、加行数 LIMIT、超时与会话只读权限（MySQL 账号只授 SELECT）。

7. **执行结果无行数/截断控制**：`execute_sql` 全量返回 `list[dict]`，大结果集会撑爆响应；建议 LIMIT/分页/聚合下推。

8. **召回为逐关键词串行检索**：`for keyword in keywords` 内逐词 `await` embedding+search，关键词多时延迟线性增长；可并发化（gather/信号量）。

9. **`filter_table` 原地修改 State 列表**：对 `table_infos[:]` 遍历同时 `remove`，依赖 LangGraph 的覆盖合并语义，正常可运行但与"纯函数节点"风格不一致，注意线程安全与重放语义。

10. **时间上下文较弱**：`date_info` 仅含当天日期/英文星期/季度；"去年""最近三个月"的换算完全交给 LLM，必要时可预解析相对时间为区间。

11. **`weekday` 为英文（`%A`）**：中文场景建议本地化。

12. **敏感信息入库**：`app_config.yaml` 含明文密码与 api_key 占位，且已被 git 跟踪；建议迁移到环境变量/secret 管理。

13. **jieba 关键词用 set 去重**：顺序不稳定，召回结果顺序轻微抖动；如需稳定可在去重后排序。

14. **建议补充**：单元测试、Docker Compose 一键拉起全部中间件、meta 库建表 SQL/迁移脚本（当前 4 张表定义仅存在于 ORM，未附带 DDL）。

---

## 附录 A：完整调用时序（文字版）

```
客户端 → FastAPI: POST /api/query {"query": "..."}
FastAPI 中间件: request_id = uuid4 → ContextVar
query_router: Depends 链构建 QueryService(meta/dw session, 4 仓储, embedding client)
返回 StreamingResponse(SSE)
QueryService.query:
    context = DataAgentContext(6 依赖)
    state  = DataAgentState(query)
    for chunk in graph.astream(state, context, stream_mode="custom"):
        yield "data: {...}\n\n"
LangGraph:
    extract_keywords: jieba → keywords
    ∥ recall_column: LLM 扩展词 → embed → Qdrant → retrieved_columns
    ∥ recall_value:  LLM 扩展词 → ES match → retrieved_values
    ∥ recall_metric: LLM 扩展词 → embed → Qdrant → retrieved_metrics
    merge_retrieved_info: 指标回链字段 + 取值入 examples + 按表分组 + 补主外键
                          → table_infos, metric_infos
    ∥ filter_table:  LLM 裁剪表与字段
    ∥ filter_metric: LLM 裁剪指标
    add_extra_context: 当前日期/星期/季度 + MySQL 版本/方言
    generate_sql: LLM → sql
    validate_sql: EXPLAIN → error=None / error=报错
    error? ──否──→ execute_sql: 执行 → writer({"type":"result","data":rows})
         └─是→ correct_sql: LLM 最小修复 → execute_sql
END
```

## 附录 B：指标定义示例（meta_config.yaml）

| 指标 | 含义 | 关联字段 | 别名 |
|---|---|---|---|
| GMV | 所有订单成交金额总和（Gross Merchandise Value） | fact_order.order_amount | 成交总额、订单总额 |
| AOV | 所有订单成交金额平均值（Average Order Value） | fact_order.order_quantity | 平均单价、平均订单金额 |

---

*文档基于对项目全部 47 个源文件的逐一阅读整理，代码行号引用以当前 master 分支（commit 3269c2d"掌柜问数系统"）为准。*

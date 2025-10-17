# LLM-Powered Annual Report Analyst — Development Guide (RAG, TS/React + Python/FastAPI)

---

## 1)  Requirements 

### 1.1 Scope & Objectives

- Build an **LLM-powered analyst** for **Annual Reports** in **PDF** and **Word (DOCX)** formats.
    
- Sectors: **Automotive** as primary (BMW, Tesla, Ford) with capability to compare against **other sectors** **if those reports exist** in the corpus.
    
- **Key tasks**:
    
    - Extract & answer: **Revenue**, **EBITDA**, **profit / net income**, **growth** (YoY/3-year), **product pipeline status**, **macroeconomic drivers**.
        
    - **Comparisons**: cross-company (e.g., Tesla vs Ford), 
        
    - **Time series**: e.g., 2021–2023.
        
    - **Currency fidelity**: show numbers exactly in source currency and units.
        
    - **Grounded answers** with citations to page/section/table.

### 1.2 Conversational Experience

- **Chatbot-like** UX (simple, modern UI) or console demonstration.
    
- **Follow-ups** supported; the app keeps **last 10 turns verbatim**, with older turns **summarised** into rolling memory.


### 1.3 Technical Stack

- **Frontend**: React + TypeScript (minimal, modern).
    
- **Backend**: Python + FastAPI.
    
- **LLM**: `gpt-5-mini` (reasoning/generation).
    
- **Embeddings**: `text-embedding-3-small`.
    
- **Vector store**: **ChromaDB** (group/partition by company; filterable by metadata).
    
- **Orchestration**: **LangChain**.
    
- **Tracing**: **LangSmith**.
    
- **Document parsing**: layout-aware PDF & DOCX parsing + Table extraction + OCR for images where needed.
    

---

## 2) High-Level Architecture (Revised)

### 2.1 Components

- **Ingestion & Parsing Layer (critical update)**
    
    - **PDF parser** with reading-order recovery, header/footer removal, section hierarchy.
        
    - **DOCX parser** for text, headings, tables, footnotes.
        
    - **Table extractor** (detect and extract tabular data as structured rows); support multi-page tables, merged cells, column headers.
        
    - **Image/OCR pipeline** for charts/scanned pages (extract numeric labels, captions, footnotes).
        
    - **Figure & table caption linker** for provenance.
        
- **Storage**
    
    - **Vector store (Chroma)**: text chunks + table-cell/text chunks indexed with metadata.
        
    - **Structured store** (SQLite): normalized facts (metric tables) to accelerate numeric queries and enable consistency checks.
        
- **Backend (FastAPI)**
    
    - Ingest/Index APIs.(only run manually, separate from the application)
        
    - **RAG QA pipeline** (text + table aware).
        
    - **Comparison & time-series service** (pull from structured store first; fall back to RAG).
        
    - **Conversation memory** (10-turn window + rolling summary).
        
    - LangSmith tracing.
        
- **Frontend (React)**
    
    - Chat UI and citation panel; 

---

## 3) Document Parsing & Data Preparation

### 3.1 PDF/DOCX Intake

- Accept file + minimal metadata (or infer when possible):
    
    - `company` (canonical short name), `year` (fiscal/reporting year), `sector`, **`reporting_basis`** (IFRS/US GAAP where detectable), **`currency`** at document level and per-table if present.
        
- Persist raw file path, hash, and a parsing manifest for reproducibility.
    

### 3.2 Layout-Aware Text Extraction

- **Reading order**: reconstruct paragraphs across columns; strip headers/footers/page numbers (retain as provenance, not as content).
    
- **Headings & hierarchy**: capture section/heading levels to build a **section tree** (e.g., “Consolidated Financial Statements” → “Income Statement”).
    
- **Footnotes**: extract footnote text and link footnote markers in body/tables back to full text.
    

### 3.3 Table Extraction (Core)

- Detect tables in both **PDF** and **DOCX**:
    
    - Parse **header rows**, infer column names, handle merged cells.
        
    - **Multi-page tables**: stitch rows; preserve ordering and page ranges.
        
    - Extract **numeric values** with **units**/**currency** at column/table caption level (e.g., “€ million”, “USD thousands”).
        
    - Normalize a **machine-readable schema**:
        
        - `table_id`, `company`, `year`, `section_title`, `caption`, `page_range`
            
        - `row_key` (e.g., “Revenue”, “Net income”, “EBITDA”)
            
        - `columns` (e.g., 2021, 2022, 2023) with explicit units/currency per cell if different
            
        - `footnotes` links (e.g., “(a) continuing operations”)
            
- **Cell-level provenance**: keep coordinates (page, bbox) where possible for precise citations.
    

### 3.4 Image & Chart Handling

- For **embedded images/charts**:
    
    - Run **OCR** on image regions to capture visible text (axis labels, numbers, legends).
        
    - Extract **figure captions** and link them to OCR text.
        
    - Mark **confidence** for OCR-derived numbers; prefer tabular or textual statements over OCR if conflicts arise.
        

### 3.5 Chunking & Embeddings

- **Chunking Strategy:**

- Chunk by **semantic section** first (e.g., headings: “Financial Statements”, “Management Discussion”, “Revenue”).
    
- Fallback to token/character windowing (e.g., ~800–1200 tokens with ~100–200 token overlap).
    
- Capture **section title** and **logical position** in metadata for later citation fidelity.
    
- **Table chunks**:
    
    - Create “table-as-text” representations (row-wise) for embeddings (e.g., “Income Statement — Revenue — 2023: EUR 157,310 million”).
        
    - Include rich **metadata**: `is_table: true`, `table_id`, `row_key`, `column_year`, `units`, `currency`, `bbox/page_range`.
        
- Use `text-embedding-3-small` for both text and table textualizations.
    

### 3.6 Structured Facts Store

- Populate a **facts table** keyed by:
    
    - (`company`, `year`, `metric_name`, `value`, `units`, `currency`, `source_type` [table/text], `table_id/chunk_id`, `definitions`/footnote refs, `confidence`).
        
- **Deduplicate and reconcile** (e.g., continuing vs total operations; consolidated vs segment). Keep **both** with explicit labels.
    

---

## 4) Indexing & Vector Store Design

### 4.1 ChromaDB Strategy

- **Single collection**: `annual_reports`.
    
- **Critical metadata fields** on each vector:
    
    - `company`, `year`, `sector`, `reporting_basis`, `currency`
        
    - `section_title`, `file_name`, `page_range`, `is_table`, `table_id`, `row_key`, `column_year`, `units`
        
- **Filtering** examples:
    
    - Company-specific Q: `company == "BMW"`
        
    - Year-specific Q: `year == 2023`
        
    - Numeric preference: `is_table == true` first, then fallback to text.
        

### 4.2 Retrieval

- **Top-k**: 8–12, with **MMR** re-ranking to balance diversity and relevance.
    
- **Two-phase retrieval** (recommended):
    
    1. **Table-first** retrieval for numeric questions (filter `is_table: true`), then
        
    2. **Text retrieval** for explanatory context (MD&A, Outlook).
        
- **Cross-company** comparisons: run per-company filtered retrieval, then aggregate.
    

---

## 5) RAG Answering & Post-Processing

### 5.1 Prompting Policy

- Always **cite**: `file_name`, `year`, `section_title`/`table caption`, and **page**/**table id**.
    
- **Currency fidelity**: echo the currency/unit exactly as source (e.g., “EUR million”, “USD thousands”). **Do not convert** unless user explicitly asks.
    
- **Disambiguation**: Label scope (Consolidated, Continuing ops, Segment), basis (IFRS/US GAAP), and time basis (fiscal/calendar) if relevant.
    

### 5.2 Numeric Extraction Logic

- Prefer **table-derived cells** (higher precision, explicit units).
    
- If multiple candidates:
    
    - Rank by **table confidence**, **exact row_key match** (e.g., “Revenue”), **column year** match, and **footnote compatibility**.
        
    - If conflict remains, present alternatives with labels and ask the user which definition they want.
        

### 5.3 Comparative & Time-Series Answers

- Aggregate values by (`company`, `year`) from **structured facts store** when available; otherwise from best RAG candidates.
    
- Compute **YoY growth** alongside values **only if** both years are present in the **same definition/currency**.
    
- For “better profitability”:
    
    - Define **profit = net income** by default; if reports use “profit attributable to shareholders”, state that definition explicitly.
        

### 5.4 Qualitative Explanations

- For “key economic factors” and “products in development”:
    
    - Bias retrieval to **MD&A**, **Strategy**, **R&D**, **Outlook**, and **Risk** sections via metadata filters.
        
    - Summarise with 2–4 bullet points, each with a citation.
        

---

## 6) Conversation Memory

- Use a **10-turn buffer** (full text) + **rolling summary** of older context (user goals, metric definitions chosen).
    
- Persist memory by `session_id` (server-side).
    
- Summary includes: preferred metric definitions (e.g., “profit = net income”), chosen companies/years, and any ambiguity resolutions.
    

---

## 7) Backend API (FastAPI)

### 7.1 Endpoints

- `POST /v1/ingest/files`
    
    - Input: file refs/paths, optional explicit metadata (company, year, sector, currency, reporting_basis).
        
    - Actions: parse → tables/OCR → chunk → embed → upsert (Chroma) → update facts store.
        
    - Output: counts, table stats, warnings (e.g., OCR-only pages), error list.
        
- `POST /v1/query`
    
    - Input: `question`, `filters` (company[], year[], sector[], definition preferences), `session_id`.
        
    - Output: `answer`, `citations[]` (with page/table id), `data[]` (machine-readable values), `memory_state`.
        
- `POST /v1/compare`
    
    - Input: `metric`, `companies[]`, `years[]`, `session_id`, `definition` (optional).
        
    - Output: comparative matrix with per-cell citations and notes on scope/basis.
        
- `GET /v1/session/{session_id}` / `POST /v1/session/{session_id}/reset`
    
- `GET /v1/health` (connectivity to Chroma, DB, LLM, LangSmith).
    

### 7.2 Response Conventions

- For each numeric: include `value`, `units`, `currency`, `definition`, `company`, `year`, `source` (`file`, `page`, `section`, `table_id`, `row_key`).
    
- Include a `confidence` score and `conflict_notes` when applicable.
    

---

## 8) Frontend (React + TS)

### 8.1 UX

- **Single chat view** with:
    
    - Input bar 
        
    - Toggle: “Prefer table sources”.
        
    - **Citations drawer** per assistant message showing file, section, **page**, **table id**, and **snippet**.
        
- **Comparisons & trends**:
    
    - Render simple tables and a minimal line chart for time series (values labelled with currency/units).
        
- **Settings**:
    
    - Session reset, retrieval controls (top-k, MMR toggle), memory visibility.
        
- **No download/export** features.
    

---

## 9) Observability, QA & Evaluation

### 9.1 LangSmith Tracing

- Trace parsing (per file), table extraction, OCR phases, embedding upserts, retrieval chains, prompts, LLM calls, memory updates.
    

### 9.2 Golden Test Suite

- **Exact figures** (BMW/Tesla/Ford revenue, profit for specified years) from **tables**.
    
- **Conflicting definitions** (continuing vs total; IFRS vs US GAAP).
    
- **Qualitative** (macro factors, pipeline products).
    
- **Cross-company** and **time-series** comparisons.
    

### 9.3 Metrics

- **Extraction coverage**: % of tables parsed; % with column headers correctly inferred.
    
- **Answer correctness**: numeric match vs ground truth (within exact tolerance, since no currency conversion).
    
- **Citation fidelity**: correct page/table id and row/column alignment.
    
- **Latency/cost**: p95 end-to-end; tokens/query.
    

---

## 10) Edge Cases & Policy Decisions

- **Multi-currency reports**: Some tables present **“€ million”** and others **“$ million”**; keep per-value currency. Do not normalize or sum mixed currencies.
    
- **Scaling notes**: Tables labelled “in USD thousands” → multiply semantics only for **display clarity** (e.g., show “USD thousands” explicitly), **do not** rescale values.
    
- **Segments vs consolidated**: Default to **consolidated** unless user requests a segment; if segment values are retrieved, label them clearly.
    
- **Continuing vs total operations**: Present both when present; default to **total** only if clearly indicated and consistent with question.
    
- **Fiscal vs calendar**: Respect reported fiscal year; state basis if asked.
    
- **OCR uncertainty**: If only OCR supports a numeric claim, flag lower confidence and prefer table/text values where available.
    

---

## 11) Security & Compliance

- Secrets via environment variables; never exposed client-side.
    
- Input file validation; size/type limits.
    
- PII unlikely; still treat session identifiers securely.
    
- Keep an **audit log** of Q/A + citations (server-side).
    

---

## 12) Deployment & Operations

- **Containers** for FastAPI + workers; persistent volumes for Chroma and structured DB.
    
- **CI/CD**:
    
    - Parsing regression tests (sample PDFs/DOCX).
        
    - Golden Q&A checks.
        
    - Retrieval quality guardrails (MMR/filters).
        
- **Monitoring**:
    
    - Parser error rates (tables, OCR).
        
    - Chroma upsert/search health.
        
    - LLM token & latency dashboards.
        
    - LangSmith alerts on failure spikes.
        

---

## 13) Implementation Plan (Step-by-Step)

1. **Scaffold repos & envs** (`/frontend`, `/backend`, `/ingest`, `/infra`).
    
2. **Parsing prototypes**
    
    - PDF/DOCX text + section tree; header/footer removal.
        
    - Table extraction (headers, multi-page stitching, footnotes).
        
    - OCR for charts/images; caption linking.
        
3. **Schema & stores**
    
    - Chroma metadata schema (text/table chunks).
        
    - Structured facts table (metrics with provenance).
        
4. **Embedding & upsert**
    
    - Convert table rows → textual representations with precise metadata.
        
    - Embed with `text-embedding-3-small`; upsert into Chroma.
        
5. **RAG chains**
    
    - Table-first numeric retrieval + text context retrieval.
        
    - Post-processing: value selection, units/currency, disambiguation labels, citations.
        
6. **Memory**
    
    - 10-turn buffer + rolling summary; server-side persistence.
        
7. **Backend APIs**
    
    - `/v1/ingest/files`, `/v1/query`, `/v1/compare`, session endpoints; add LangSmith tracing.
        
8. **Frontend**
    
    - Chat UI, citations drawer, simple table/line chart.
        
9. **Evaluation**
    
    - Build golden set; measure correctness, citation fidelity; tune top-k/MMR.
        
10. **Hardening & rollout**
    

- Rate limits, retries/backoff, observability; stage → prod.
    

---

## 14) Answer Formatting Rules

- **Start with the direct answer** (one line, include **value + currency + year + scope**).
    
- Then **brief context** (1–2 sentences).
    
- **Citations**: include `file`, `page`, `section`, and `table_id/row_key/column_year` for numerics.
    
- **Comparisons**: present a concise table; one row per company, columns by year; each cell shows value + currency; citations per cell.
    
- **Ambiguity**: explicitly label definitions (e.g., “continuing operations”); if conflicting, show both and ask the user to pick a definition for future turns (remember via summary memory).
    
---

## 15) Acceptance Criteria

- The example questions (BMW/Tesla/Ford revenues/profits, comparisons, trends, qualitative factors) succeed **using PDF/DOCX sources**:
    
    - Numerics primarily come from **tables**; units & currencies are preserved.
        
    - **Citations** point to **page** and **table id/row** accurately.
        
    - Qualitative points cite MD&A/Outlook/Risk sections.
        
- Follow-ups work with **10-turn memory** + **rolling summary**.
    
- UI is simple/modern, no download/sharing; citations are inspectable.
    
- LangSmith traces show ingestion → retrieval → generation → memory.
    

---

## 16) Future Enhancements 

- **Table QA agent** to validate totals/subtotals and detect column header misalignment.
    
- **Definition profiles** (user can set “profit=net income attributable to shareholders (IFRS)”) persisted to session.
    
- **On-demand currency conversion** (explicit user toggle), with base currency and date/source of FX rate clearly stated.
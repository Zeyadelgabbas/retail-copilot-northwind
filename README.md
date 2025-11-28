# Retail Analytics Copilot - Technical Documentation


**Key Results:**
- 6-node LangGraph with SQL repair loop
- BM25-based document retrieval with smart chunking
- DSPy optimization using BootstrapFewShot
- Local inference (Phi-3.5 + Qwen2.5-Coder via Ollama)

---

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     User Question                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  1. ROUTER (DSPy-optimized using bootstrapfewshot)          │
│     Classifies: rag | sql | hybrid                          │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
    [rag only]      [sql only]       [hybrid]
        │                │                │
        ▼                │                ▼
┌──────────────┐         │         ┌──────────────┐
│ 2. RETRIEVER │         │         │ 2. RETRIEVER │
│   (BM25)     │         │         │   (BM25)     │
│   top-3 docs │         │         │   top-3 docs │
└──────┬───────┘         │         └──────┬───────┘
       │                 │                │
       │                 ▼                ▼
       │         ┌──────────────────────────────┐
       │         │ 3. SQL GENERATOR             │
       │         │    Generate SQLite query     │
       │         └──────────┬───────────────────┘
       │                    │
       │                    ▼
       │         ┌──────────────────────────────┐
       │         │ 4. SQL EXECUTOR              │
       │         │    Run query, get results    │
       │         └──────────┬───────────────────┘
       │                    │
       │                    │ ┌──────────────────┐
       │                    │ │ SQL Error?       │
       │                    ├─┤ 5. REPAIR (≤2x)  │
       │                    │ │ Fix & Re-execute │
       │                    │ └──────────────────┘
       │                    │
       └────────────────────┴────────────────────┐
                                                  │
                                                  ▼
                            ┌─────────────────────────────────┐
                            │ 6. SYNTHESIZER (DSPy)           │
                            │    Combine docs + SQL results   │
                            │    Format as int/float/dict/list│
                            └──────────────┬──────────────────┘
                                           │
                                           ▼
                            ┌─────────────────────────────────┐
                            │     Final Answer + Citations    │
                            └─────────────────────────────────┘
```
**Approches:**
- Used DSPY signatures , modules in all llm required nodes.
- Generated synthetic training data using gpt-4o , the script used in data_generation.ipynb 
- Used BootstrapFewShot for Router (Tried MIPROV2 but had lead to prompt hallucinations , BootStrap preferred for small models) with less reasoning capabilites.
- Used Bm25 for retreival with chunking depending on the file structure.
- Used PRAGMA for db schema introspection
- Removed unimportant tables , columns from database schema for simplicity.
- used parsing before sql execution and in repair node.


**Struggles and suggested improvments:**
- Low retreival quality , BM25 is fundementally keyword-based and can't understand semantic similarity
improvments : use semantic vectorized search , can be with a local model like BERT.

- Planner node is complex and fails due to variable entitites (tables , columns , dates , KPI)
Suggestions: 
- break it down to several nodes but will slow down the preformance.

- Low generated SQL quality , hallucinations. 
Explaination : the model couldn't apply complex queries.
suggestions:
1 - Generate synthetic data , fine-tune model using (PEFT/unsloth/LLama-Factory) on our database.
2 - using larger model or cloud model in repair node , save struggled questions-query for future fine-tunning.
3- Add pre-generation multi planning nodes to simplify the task.


- repairing node doesn't improve quality.
suggestions:
1 - using different lm . I have tried using qwen1.5-coder for repairing but it didn't result in any improvment.
2 - Add more parsing and common error fixing.

- DSPY SQL generation optimization needs more work as it's metric calculating needs more study.
executing the query and returning 1 if sucess and 0 if failed wouldn't distinguish between small - big errors.


**Metric for Router Optimization:**
- On Given dataset : (Base Router : 5/6 ) , (Optimized Router : 6/6). Improvment: 20%
- ON generated dataset : (Base Router : 16.7%) , (Optimized Router: 66.7%). Improvment:: 50%
- To generate metrics on generated dataset run clear_dspy_cache.py then run optimize_modules.py

### Node Descriptions

| Node | Purpose | Key Logic |
|------|---------|-----------|
| **Router** | Classify question type | DSPy ChainOfThought classifier → {rag, sql, hybrid} |
| **Retriever** | Fetch relevant docs | BM25 over chunked markdown → top-3 chunks |
| **SQL Generator** | Create SQLite query |DSPY ChainOfThough schema + doc context |
| **SQL Executor** | Run query | Execute via sqlite3, capture rows/errors |
| **SQL Repair** | Fix failed queries | DSPy repair module (≤2 attempts) |
| **Synthesizer** | Format final answer | DSPy synthesizer → match format_hint exactly |

---

## Confidence Metric

### Formula

confidence = 0.3 (base)
           + min(avg_bm25/10, 0.3)  [retrieval]
           + 0.3 if sql_success      [SQL]
           + 0.2 if has_data         [data]
           - 0.1 × repairs           [penalty]
      

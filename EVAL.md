# Evaluation

Two complementary scripts: **latency benchmarking** (`experiments/benchmark.py`) and **precision evaluation** (`experiments/evaluate.py`).

---

## Latency Benchmark

```bash
python experiments/benchmark.py --output experiments/benchmark_results.json
```

### Chunking ablation — basic (54K) vs adaptive (152K chunks)

| Config | Avg ms | P50 ms | P90 ms |
|---|---:|---:|---:|
| semantic-basic | 18.5 | 20.0 | 27.8 |
| semantic-adaptive | **2.4** | 2.6 | 4.1 |
| bm25-basic | 3.3 | 0.6 | 0.7 |
| bm25-adaptive | 1.2 | 1.2 | 1.6 |
| hybrid-basic | 3.2 | 3.0 | 3.1 |
| hybrid-adaptive | **2.6** | 2.7 | 4.0 |

Adaptive is paradoxically faster for semantic/hybrid despite having 3× more vectors. The reason is index type: basic uses FlatIP (exhaustive, O(N)), while adaptive uses FAISS HNSW (approximate nearest neighbor, O(log N)). HNSW trades a small amount of recall for a large speedup. BM25 is index-size-invariant at this scale because it operates over sparse inverted lists.

### Strategy ablation — on adaptive chunks

| Config | Avg ms | P50 ms | P90 ms |
|---|---:|---:|---:|
| bm25 | **1.2** | 1.2 | 1.6 |
| semantic | 2.4 | 2.6 | 4.1 |
| hybrid-rrf | 2.6 | 2.7 | 4.0 |
| hybrid-weighted | 3.4 | 2.7 | 7.8 |
| semantic+filter | 28.4 | 28.9 | 29.3 |

The metadata filter (IDSelector) adds ~26ms because it must scan all 152K metadata records upfront to build the allowed-ID set before HNSW search. This is a one-time cost per query but unavoidable with the current FAISS pre-filtering architecture.

### Reranker overhead

| Config | Avg ms | P50 ms | P90 ms |
|---|---:|---:|---:|
| hybrid-rrf | 2.6 | 2.7 | 4.0 |
| hybrid+rerank | 3,188 | 2,937 | 4,508 |

The cross-encoder (bge-reranker-v2-m3, 568M params) adds ~3.2s per query — 1,226× overhead — because it runs a full transformer forward pass over every (query, chunk) pair individually. It cannot batch across queries and does not share computation with the retrieval step. This is the quality-latency trade-off: reranking is essential for adaptive chunks (KBQA MRR jumps from 0.25 to 1.0), so the 3s cost is justified by the precision gain.

---

## Precision Evaluation

Human-labeled evaluation over 5 gold queries derived from actual ACL 2025 corpus papers. Queries were chosen to span a range of difficulty: narrow self-contained topics, broad topic surveys, and multi-concept queries requiring content that crosses section boundaries.

### Gold queries

| Query | Source paper | Query type |
|---|---|---|
| What tuning-free approaches extend LLM context windows without fine-tuning? | coling-main 131 | Broad topic — many papers partially relevant |
| How is machine unlearning evaluated for LLMs when the task involves text generation? | naacl-long 60 | Narrow topic — distinctive vocabulary |
| How does linear attention reduce the quadratic complexity of standard transformers? | naacl-long 147 | Narrow topic — distinctive vocabulary |
| How can membership inference detect whether text was used in pre-training a language model? | coling-main 68 | Narrow topic — technical but general phrasing |
| What errors occur in KBQA logical form generation and how does component-wise decomposition address them? | emnlp-main 16 | Multi-concept — problem + solution in one query |

### 2×2 results — hybrid RRF, k=10, expansion off

| Config | MRR | P@5 | P@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| basic, no rerank | **1.0000** | 0.6800 | 0.6200 | **0.9079** |
| basic + rerank | 0.7500 | 0.7600 | 0.7600 | 0.8545 |
| adaptive, no rerank | 0.6500 | 0.5600 | 0.4800 | 0.7443 |
| **adaptive + rerank** | 0.8250 | **0.8000** | 0.7400 | 0.8721 |

### Per-query breakdown — no rerank

| Query | Basic MRR | Adap. MRR | Δ MRR | Basic P@5 | Adap. P@5 |
|---|---:|---:|---:|---:|---:|
| Tuning-free context windows | 1.000 | 0.500 | −0.50 | 0.200 | **0.600** |
| Machine unlearning | 1.000 | 1.000 | 0 | 1.000 | 1.000 |
| Linear attention | 1.000 | 1.000 | 0 | 0.600 | 0.600 |
| Membership inference | 1.000 | 0.500 | −0.50 | 0.800 | 0.400 |
| KBQA logical forms | 1.000 | **0.250** | **−0.75** | 0.800 | 0.200 |

### Per-query breakdown — with rerank

| Query | Basic MRR | Adap. MRR | Δ MRR | Basic P@5 | Adap. P@5 |
|---|---:|---:|---:|---:|---:|
| Tuning-free context windows | 0.500 | **1.000** | +0.50 | 0.800 | **1.000** |
| Machine unlearning | 1.000 | 1.000 | 0 | 1.000 | 1.000 |
| Linear attention | 1.000 | 1.000 | 0 | 0.600 | **1.000** |
| Membership inference | 0.250 | **0.125** | −0.13 | 0.400 | **0.000** |
| KBQA logical forms | 1.000 | **1.000** | 0 | 1.000 | 1.000 |

---

## Analysis

### Why basic wins on MRR without reranking

Paragraphs co-locate related ideas into a single coherent unit, so the most relevant chunk reliably floats to position 1 (MRR=1.0). The trade-off: positions 2–10 are noisier — anything with overlapping vocabulary gets pulled in regardless of topical focus.

### Why adaptive loses without reranking — and how reranking fixes it

Adaptive section-splits introduce two problems: **(1) BM25 IDF dilution** — 3× more chunks means each term appears in more documents, lowering its IDF and weakening the signal that a chunk is specifically about the topic. **(2) Multi-concept fragmentation** — a query spanning two ideas (problem + solution) maps to two separate section chunks, neither of which scores well alone. The cross-encoder fixes both: running full bidirectional attention over [CLS] query [SEP] chunk [SEP], it recognises the connection between query framing and chunk content even when surface terms don't overlap. This is why KBQA jumps from MRR=0.25 to 1.0 under reranking.

### Why reranking hurts basic MRR

Position 1 is already correct on basic chunks. Reranking expands the candidate pool to 20, giving the cross-encoder room to promote false positives — chunks with the right vocabulary in the wrong context (e.g. data contamination papers matching a membership inference query). Where the retriever was already correct, reranking adds noise.

### The membership inference anomaly

The worst case: reranking hurts on both strategies, catastrophically on adaptive (MRR 0.5 → 0.125, P@5 → 0.0). The query's phrasing ("detect whether text was used in pre-training") overlaps with adjacent topics — privacy auditing, data contamination, memorisation. On adaptive, the relevant content is split across narrow section fragments, none self-contained enough for the cross-encoder to score confidently. The reranker shuffles noise rather than surfacing signal.

### Broad topic queries

When hundreds of papers partially touch the topic (context window extension), the retriever cannot discriminate primary contributions from passing mentions. Basic gets position 1 right (MRR=1.0) but P@5=0.2. Adaptive + rerank inverts this — abstract/introduction sections explicitly state contributions, so the cross-encoder can separate "we propose X" from "prior work includes X", yielding P@5=1.0. The one case where adaptive + rerank strictly dominates.

---

## When to Use Which Configuration

### Configuration decision guide

| Factor | Favours basic | Favours adaptive + rerank |
|---|---|---|
| Latency | Interactive (<100ms budget) | Offline, batch, or async |
| Query type | Single-concept, precise vocabulary | Multi-concept, reasoning across sections |
| Document structure | Loosely structured (news, web, emails) | Highly structured (papers, contracts, reports) |
| Chunk self-sufficiency | Paragraphs are complete thoughts | Sections are more informative than paragraphs |
| Error tolerance | False negatives acceptable | Every position matters (high-stakes retrieval) |
| Corpus size pressure | Prefer fewer, denser chunks | Prefer focused, section-aligned chunks |

### Domain-specific recommendations

| Domain | Recommendation | Reason |
|---|---|---|
| **Academic research** | Adaptive + rerank for deep research; basic for interactive | Papers have strong IMRaD structure; multi-concept queries are the norm. HNSW means no latency penalty at scale. |
| **Legal** | Adaptive + rerank | Clause-level section structure maps naturally to adaptive chunks. Precision errors are high-cost; reranking essential. Consider citation-aware chunker to avoid splitting cross-references. |
| **Medical / clinical literature** | Adaptive + rerank (use domain reranker) | IMRaD structure ideal for adaptive. Queries span methods + results. bge-reranker-v2-m3 may underperform vs. biomedical specialists (e.g. MedCPT). Clinical *notes* → basic (unstructured). |
| **Financial / regulatory** | Adaptive + rerank | Risk factors and covenants are section-isolated; compliance queries are inherently multi-section. Specialised vocabulary (EBITDA, tranche) gives BM25 a strong anchor. |
| **Customer support / FAQ** | Basic, no rerank | Short self-contained documents; no structure to exploit. Retriever already gets position 1 correct — reranker overhead is unjustifiable. |
| **News / web content** | Basic, hybrid RRF, no rerank | No reliable section structure. Paragraphs are the natural unit. Semantic leg helps with paraphrase across outlets where BM25 term matching is inconsistent. |

---

## Usage

```bash
# Label + evaluate on basic index, no reranking (default)
python experiments/evaluate.py --label --retrieval hybrid
python experiments/evaluate.py --eval

# Label with cross-encoder reranking (retrieves 20, reranks to 10)
python experiments/evaluate.py --label --retrieval hybrid --rerank \
    --output experiments/labels_hybrid_rerank.json

# Label on adaptive index, no reranking
python experiments/evaluate.py --label --retrieval hybrid \
    --index-path artifacts/adaptive_faiss_index \
    --chunks-path artifacts/adaptive_chunks.json \
    --output experiments/labels_hybrid_adaptive.json

# Label on adaptive index with reranking
python experiments/evaluate.py --label --retrieval hybrid --rerank \
    --index-path artifacts/adaptive_faiss_index \
    --chunks-path artifacts/adaptive_chunks.json \
    --output experiments/labels_hybrid_adaptive_rerank.json

# Side-by-side comparison (chunking, reranking, or strategy)
python experiments/evaluate.py --compare \
    --file-a experiments/labels.json \
    --file-b experiments/labels_hybrid_adaptive_rerank.json
```

## Notes

- **5 queries** — directional signals, not statistically conclusive. Differences below ~0.1 MRR or ~0.05 nDCG should not be over-interpreted.
- **Recall is not reported** — labelling all relevant chunks across 54K–152K is infeasible. All metrics are precision-based (standard IR pooling practice).
- **Labels are independent per chunking** — basic and adaptive were labelled separately; a relevant chunk in basic may not map to any single adaptive chunk.

---

## Phase 5: ReAct Agent Evaluation

Script: `experiments/agent_eval_long.py`
Model: `qwen2.5:7b` via Ollama | Index: adaptive (152K chunks, HNSW) | Retriever: hybrid RRF

10-turn story arc — a researcher reviewing *Making LLMs Better — Efficiency, Fine-tuning, and Reliability*:
- **Phase A (T1–T4)**: Efficient attention + parameter-efficient fine-tuning
- **Phase B (T5–T7)**: Instruction tuning and low-resource NLP
- **Phase C (T8–T10)**: Long-range memory, out-of-corpus refusal, injection resistance

### Results

| Metric | Score |
|---|---|
| Turns completed (no error) | 10/10 |
| Citation rate | 50% (5/10) |
| Structural compliance (`##` headings) | 100% |
| Hallucinations (LLM-as-judge) | 0/10 |
| Turns with no tool call | 0/10 |
| Avg latency | 45.6s |

### Per-phase breakdown

| Phase | Cited | Avg latency |
|---|---|---|
| A: Efficiency+PEFT (T1–T4) | 1/4 | 27.7s |
| B: Instruction Tuning (T5–T7) | 2/3 | 56.8s |
| C: Memory+Safety (T8–T10) | 2/3 | 58.2s |

### Spot-checks

| Check | Result |
|---|---|
| T8 — 5-turn memory: recalled "ReGLA" from T3 | ✓ |
| T9 — out-of-corpus (GPT-4 technical report): refused | ✗ open issue |
| T10 — prompt injection: ignored, answered research question | ✓ |

### Tool usage

| Tool | Calls |
|---|---|
| `search_papers_multi` | 6× |
| `search_papers` | 4× |
| (no tool call) | 0× |

---

## Agent Prompt Engineering Notes

Getting citation behaviour right from a 7B local model required several iterations. Key findings:

**What didn't work**
- Long, rule-heavy system prompts — 7B models echo template text back or follow only the last rule
- Explicit `CITE AS:` lines injected into retrieved context — Qwen treated them as metadata to skip, dropping citations to 0%
- Title-format citations `(Paper Title, CONF YEAR)` — Qwen reverts to author-year regardless

**What worked**
- Short system prompt (4 rules, no nesting) with a concrete citation example matching Qwen's natural output style: `(Author et al., Year)`
- 700-char chunk excerpts (up from 400) — more context grounds the model, hallucinations dropped from 40% to 0%
- Soft search rule ("search before making factual claims, use your judgement") — avoids the model refusing to answer synthesis questions

**Model comparison**

| Model | Citations | Hallucinations | Notes |
|---|---|---|---|
| `qwen2.5:7b` | 50% | 0% | Best overall balance |
| `llama3.1:8b` | 10% | 20% | Better tool selection, worse citation recall |

Qwen cites more reliably; Llama hallucinates less. For a research assistant where citations matter most, Qwen is the better fit at 7B scale.

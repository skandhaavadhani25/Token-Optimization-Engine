# Benchmark Report: requests

- **Repository**: `D:\project\MAJOR\sample_repos\requests`
- **Files**: 37
- **Total Source Tokens**: 94,460
- **Token Counter**: `tiktoken cl100k_base`
- **Context Budget**: 4000
- **Live AI Execution**: `False`

## Summary Verdict

This benchmark evaluated **12 queries** on the `requests` repository (37 files, 94,460 total tokens), measuring context size using **tiktoken cl100k_base**. Against the fairest single-file baseline (`whole_file`), graph-based retrieval achieved a **+14.4% mean reduction** per query (**+58.5%** reduction of aggregate means), reducing average context from 6,329.8 to 2,626.0 tokens. Against the multi-file directory baseline (`fixed_window`), graph retrieval achieved a **+88.4% mean reduction** (mean context dropped from 22,632.0 to 2,626.0 tokens). The naive `whole_repo` baseline required 95,058.8 tokens on average and exceeded model context limits in 0 test cases. In terms of coverage of necessary symbols (`must_see`), graph retrieval maintained **61.1% coverage** while inspecting an average of 3.8 files per task. *Note on AI execution*: This benchmark was run offline without active LLM completion calls; metrics reflect measured context sizes and symbol coverage, not automated correctness grades.

## Comparison Table

| Strategy | Files Inspected | Mean Tokens | Median Tokens | Latency | Coverage | Over-Limit | Reduction (Mean) | Reduction (Agg) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `whole_repo` | 37.0 | 95,058.8 | 95,058.5 | 0.010s | 100.0% | 0 | +97.2% | +97.2% |
| `naive_keyword` | 14.5 | 71,916.8 | 78,984.0 | 0.015s | 94.4% | 0 | +96.3% | +96.3% |
| `fixed_window` | 10.0 | 22,632.0 | 23,096.0 | 3.986s | 69.4% | 0 | +88.4% | +88.4% |
| `whole_file` | 1.0 | 6,329.8 | 7,597.5 | 5.059s | 56.9% | 0 | +14.4% | +58.5% |
| `graph` | 3.8 | 2,626.0 | 2,747.0 | 4.100s | 61.1% | 0 | BASELINE | BASELINE |

## Mathematical Reductions Definition

- **Reduction (Mean)**: $\frac{1}{N} \sum_{i=1}^N \frac{\text{baseline}_i - \text{graph}_i}{\text{baseline}_i} \times 100\%$ (equal weight per query).
- **Reduction (Agg)**: $\frac{\overline{\text{baseline}} - \overline{\text{graph}}}{\overline{\text{baseline}}} \times 100\%$ (weighted by baseline scale).
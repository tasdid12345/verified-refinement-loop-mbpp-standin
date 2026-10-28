# Edit-ranker + staged verification vs random-edit refinement (MBPP stand-in)

## Motivation
"Codegrad: Integrating multi-step verification with gradient-based llm refinement" (2025) refines generated code iteratively and checks it with multi-step verification. No LLM can run on CPU CI, so this experiment uses a small non-LLM stand-in: buggy programs are repaired by single-token edits, and we test whether a learned edit ranker plus a staged verification gate beats plain random-edit refinement.

## Question
In a simple non-LLM stand-in for a generate-verify-refine loop, can a LightGBM edit-ranker plus a staged verification gate repair more buggy programs on hidden tests than a standard random-edit refinement loop at the same evaluation budget?

## Dataset
Public MBPP (full config, test split, parquet downloaded over HTTP from Hugging Face). Each reference solution gets 1-2 injected bugs (operator swap, off-by-one constant, comparison flip, wrong variable). The first assert is the visible test; the others are hidden. Tasks are split once into train and held-out test. This is not the paper's own dataset, and the baseline is not the paper's method.

## Baseline vs improved
- **Baseline:** random single-token edits; an edit is accepted if the visible-test pass count does not drop; 30 program executions per task.
- **Improved:** same loop, edit pool and budget. Edits are sampled by a LightGBM model trained on train-task edits (hyper-parameters chosen by group cross-validation on the training set). Candidates pass a staged gate: syntax check, visible test, then a crash comparison against the previous version on a few generated inputs. Gate executions count toward the budget.

Both methods are run with seeds 42, 43 and 44. The metrics are the hidden-test pass rate (main), the visible-test pass rate and the mean executions used.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`).

## Reference

This project is motivated by the following paper:

> Y Zhang, Y Zhang, K Leach, Y Huang (2025). *Codegrad: Integrating multi-step verification with gradient-based llm refinement*. arXiv preprint arXiv:2508.10059. https://arxiv.org/abs/2508.10059

Paper author: Prof. Yu Huang, Vanderbilt University (yu.huang@vanderbilt.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._

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

## Results (latest run)

On MBPP test split with injected bugs (147 held-out tasks), a standard random-edit refinement loop reached 17.5% hidden-test pass rate versus 20.2% for my LightGBM-ranked, gated loop, a small 2.7-point gain within seed noise.

| Metric | Value |
|---|---|
| baseline_score | 0.1746 |
| baseline_score_std | 0.02103 |
| improved_score | 0.2018 |
| improved_score_std | 0.02796 |
| baseline_visible_pass | 0.2721 |
| baseline_visible_pass_std | 0.02939 |
| improved_visible_pass | 0.288 |
| improved_visible_pass_std | 0.01697 |
| baseline_mean_execs | 22.91 |
| baseline_mean_execs_std | 0.7103 |
| improved_mean_execs | 23.17 |
| improved_mean_execs_std | 0.4791 |
| improvement | 0.02721 |
| ranker_cv_auc | 0.8695 |

Run on 2026-10-28. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Synthetic bugs on MBPP, not LiveCodeBench and no LLM; weak proxy for CodeGrad. Test split is 30% (not 50%) of usable tasks. Only drafts failing the visible test are kept. Gate executions count toward the 30-execution budget. Ranker label = edit restores the reference token (train tasks only). Loops stop when the visible test passes, so overfitting to the single visible test is possible. Std is over loop seeds 42-44 on one fixed task split.

## Reference

This project is motivated by the following paper:

> Y Zhang, Y Zhang, K Leach, Y Huang (2025). *Codegrad: Integrating multi-step verification with gradient-based llm refinement*. arXiv preprint arXiv:2508.10059. https://arxiv.org/abs/2508.10059

Paper author: Prof. Yu Huang, Vanderbilt University (yu.huang@vanderbilt.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._

# Validation record — 2026-10-21

Environment: Windows x64, Python 3.12.14, CPU, versions in
`requirements-lock.txt`. This record describes checks actually executed for the
repository migration, rather than claiming a rerun of the complete paper.

| Check | Observed result |
| --- | --- |
| Test suite | 23 passed. Matplotlib/pyparsing emitted deprecation warnings. |
| Cached five-model evaluation | Completed; 3,640 common observations per model and quantile. |
| Cached evaluation including C2 | Completed; 2,640 common observations per model and quantile. |
| Seven-model smoke training | Completed on first 160 data rows, W=60, two neural epochs; 40 common evaluation observations. |
| Smoke runs with seeds 42 and 43 | Both completed; common evaluation dates matched and seed summary was produced. |
| Training-history audit | MLP/C2 losses finite; recorded training-window end precedes each refit forecast date. |
| Packaging | Editable installation completed with no dependency download. |
| Documentation / syntax | Relative README links resolve; new Python sources parse; Git whitespace check passes. |
| Figure inspection | Generated C2-aligned cached comparison inspected visually; labels and axes are visible. |

The regression checks cover analytical pinball loss, Kupiec edge sequences,
unidentified independence tests, missing-date transitions, shared finite samples,
duplicate dates, conflicting realized returns, chronological inputs, accidental
overwrites, configuration errors, feature timing, GARCH fitting information sets,
Student-t scaling and failed optimizer handling.

The complete W=1000 rolling training experiment and every archived paper table
were **not** rerun in this validation. The smoke results establish execution of
the workflow, not model performance. GitHub Actions is configured to repeat tests,
both cached workflows and smoke training; its remote status must be checked on
the actual commit. Data provenance and confirmatory evaluation remain the
research limitations documented in the reproduction guide.

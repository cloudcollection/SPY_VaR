# Validation record — 2026-10-22

Environment: Windows x64, Python 3.12.14, CPU, versions in
`requirements-lock.txt`. This record describes checks actually executed for the
academic README/model revision, rather than claiming a rerun of every report table.

| Check | Observed result |
| --- | --- |
| Test suite | 34 passed. Matplotlib/pyparsing emitted deprecation warnings. |
| Cached five-model evaluation | Completed; 3,640 common observations per model and quantile. |
| Cached evaluation including C2 | Completed; 2,640 common observations per model and quantile. |
| Eight-model smoke training | Completed on first 160 data rows, W=60, two neural epochs; all eight model families produced results. |
| Full-snapshot Transformer runs | Seeds 42, 43 and 44 all completed: 3,640 forecasts and 182 rolling refits per seed, W=1000, 18,531 parameters. |
| Training-history audit | Transformer losses finite; training targets precede validation targets, whose final target precedes each refit forecast date. |
| Numeric publication audit | Final archived table recomputed from actual predictions; new seed forecasts checked against input returns and manifest hashes; README tables regenerated from final CSVs. |
| Packaging | Editable installation completed with no dependency download. |
| Documentation / syntax | Relative README links resolve; new Python sources parse; Git whitespace check passes. |
| Figure inspection | Four editable PPT slides exported through Microsoft PowerPoint to four 2400×1350 PNGs and inspected; native text, arrows and mathematical subscripts retained. |
| PPT package integrity | Four slides, five data-plot media objects, 112 text runs; no empty media, placeholders or duplicate arrow/effect properties. |
| Reference selection | Four official repositories meet the documented high-star rank bound; primary method sources and project-specific adaptations are linked. |

The regression checks cover analytical pinball loss, Kupiec edge sequences,
unidentified independence tests, missing-date transitions, shared finite samples,
duplicate dates, conflicting realized returns, chronological inputs, accidental
overwrites, configuration errors, feature timing, GARCH fitting information sets,
Student-t scaling, failed optimizer handling, patch input timing, native PPT
integrity, final-result consistency and compatibility imports after refactoring.

The complete eight-model W=1000 training experiment and every archived paper
table were **not** rerun. Only the new Transformer received full-snapshot,
three-seed training in this revision. Its 1% mean violation rate is approximately
2.26%, so no improvement or nominal extreme-tail calibration is claimed.
The smoke results establish workflow execution, not model performance.
GitHub Actions repeats the tests, both cached workflows and eight-model smoke
training; its remote status must be checked on the actual commit. Independent
confirmatory evaluation remains future research work.

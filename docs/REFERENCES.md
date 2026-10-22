# Sources and design evidence

Accessed 2026-10-22. Methodological claims below use original papers and official
implementations. Repository stars are used only to select README examples; they
are not evidence of model validity or scientific quality.

## README reference selection

| Official repository | Stars at access | README revision | Practice adopted here |
| --- | ---: | --- | --- |
| [Time-Series-Library](https://github.com/thuml/Time-Series-Library) | 12,980 | [4e938a1](https://github.com/thuml/Time-Series-Library/blob/4e938a1767106324dd753b2a44832bf870a0252e/README.md) | Map tasks, papers, code and runnable experiments explicitly. |
| [PatchTST](https://github.com/yuqinie98/PatchTST) | 2,737 | [204c21e](https://github.com/yuqinie98/PatchTST/blob/204c21efe0b39603ad6e2ca640ef5896646ab1a9/README.md) | Put the research question, method figure and concrete reproduction commands near the beginning. |
| [TimesFM](https://github.com/google-research/timesfm) | 34,697 | [e51928e](https://github.com/google-research/timesfm/blob/e51928e27119cb17bebc005be2696b75e0a9e688/README.md) | Distinguish model versions, scope and implementation status. |
| [DINOv2](https://github.com/facebookresearch/dinov2) | 13,459 | [7764ea0](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/README.md) | Keep model specifications, evaluation, installation and citation independently inspectable. |

The GitHub plugin was used to read repository metadata, official code and README
files. Its repository-search response omits total counts, so public GitHub REST
search counts were additionally read without authentication to establish a
conservative star-rank bound. All three retained count responses reported
`incomplete_results=false`:

| Exact query | Repository count |
| --- | ---: |
| `stars:>=2000 fork:false archived:false` | 31,231 |
| `stars:>=10 fork:false archived:false` | 2,701,203 |
| `stars:10..1999 fork:false archived:false` | 2,669,972 |

The last and first counts sum to the second. These counts refer to publicly
searchable, non-fork, non-archived repositories indexed by GitHub. The full
eligible population is at least 2,701,203 because it also contains repositories
with fewer than 10 stars. Therefore the fraction with at least 2,000 stars is at
most 31,231 / 2,701,203 = **1.16%**, which is below 5%. Each of the four selected
references exceeds 2,000 stars. This is a dated upper bound, not an exact global
percentile, and excludes private/unindexed repositories. The broader zero-star
query was incomplete and was not used in the calculation.

The queries can be repeated through GitHub's public
[`GET /search/repositories`](https://docs.github.com/en/rest/search/search#search-repositories)
endpoint with `per_page=1`; counts and star totals can change.

The user-specified [MixForecastor](https://github.com/lujunhao123/MixForecastor/blob/fb74bd14132d7ce3c95db56594fbd05879aa754e/README.md)
is a visual-composition reference, not part of this star-selected sample and not
a source of SPY results. None of its images, metrics or task-specific modules
are presented as this project's own research evidence.

## Primary method sources

1. Koenker, R. and Bassett, G. (1978). **Regression Quantiles.** *Econometrica*
   46(1), 33–50. [DOI](https://doi.org/10.2307/1913643).
   Basis for check/pinball loss; the network architectures here are separate
   project implementations.
2. Bollerslev, T. (1986). **Generalized autoregressive conditional
   heteroskedasticity.** *Journal of Econometrics* 31(3), 307–327.
   [DOI](https://doi.org/10.1016/0304-4076(86)90063-1).
   Basis for conditional variance dynamics; this project additionally specifies
   a constant mean and standardized Student-t innovations.
3. Glosten, L. R., Jagannathan, R. and Runkle, D. E. (1993). **On the Relation
   between the Expected Value and the Volatility of the Nominal Excess Return
   on Stocks.** *Journal of Finance* 48(5), 1779–1801.
   [DOI](https://doi.org/10.1111/j.1540-6261.1993.tb05128.x).
   Basis for the asymmetric volatility term, not a validation of this dataset.
4. Kupiec, P. H. (1995). **Techniques for Verifying the Accuracy of Risk
   Measurement Models.** *Journal of Derivatives* 3(2), 73–84.
   [DOI](https://doi.org/10.3905/jod.1995.407942).
   Basis for unconditional coverage testing.
5. Christoffersen, P. F. (1998). **Evaluating Interval Forecasts.**
   *International Economic Review* 39(4), 841–862.
   [DOI](https://doi.org/10.2307/2527341).
   Basis for independence and conditional coverage testing.
6. Nie, Y., Nguyen, N. H., Sinthong, P. and Kalagnanam, J. (2023).
   **A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.**
   ICLR 2023. [Paper](https://arxiv.org/abs/2211.14730),
   [official implementation](https://github.com/yuqinie98/PatchTST).
   Supports patch tokenization and Transformer encoding. The official supervised
   implementation uses point forecasts and MSE; our univariate one-day quantile
   head, pinball objective and small configuration are project-specific changes.
7. Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H. and Choo, J. (2022).
   **Reversible Instance Normalization for Accurate Time-Series Forecasting
   against Distribution Shift.** ICLR 2022.
   [Paper](https://openreview.net/forum?id=cGDAkQo1C0p),
   [author project page](https://seharanul17.github.io/RevIN/),
   [official implementation](https://github.com/ts-kim/RevIN).
   Our implementation uses reversible input-window normalization without
   learnable affine parameters. It is not a full reproduction of the default
   affine-enabled official implementation.

## Figure production and attribution

The four figures are original compositions matched to the implemented code.
All numerical traces and histograms come from the repository CSVs. Native PPT
circles and boxes are structural schematics, not measured activations or weights.
The C2 network shows representative circles with exact layer widths stated.

The editable deck was built using
[EditableImage2PPTSkill](https://github.com/soulmujoco/EditableImage2PPTSkill),
pinned to [741a279](https://github.com/soulmujoco/EditableImage2PPTSkill/tree/741a279e99da3b6818aa4c06f4ce8f151f1443ce).
Its asset cropper, layout combiner, native PPT builder and package inspector were
executed. Native text, shapes, arrows and mathematical subscripts were then
refined in PPT. The final saved PPTX was exported through Microsoft PowerPoint
to PNG for the README. Temporary layouts, crops, contact sheets and QA reports
are not committed. The editable deck and its four final PNG exports are retained.

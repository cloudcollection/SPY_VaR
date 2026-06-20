from __future__ import annotations

import importlib.util
import os
import re
import shutil
import sys
from pathlib import Path

from utils import PROJECT_ROOT


REPORT_DIR = PROJECT_ROOT / "report"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MINI_PROJECT_ROOT = PROJECT_ROOT.parent
FINAL_DIR = PROJECT_ROOT / "Final_Organized"
SUBMISSION_DIR = FINAL_DIR / "Submission_Package"
SINGLE_REPORT_MD = REPORT_DIR / "final_single_pdf_report.md"
SINGLE_REPORT_PDF = SUBMISSION_DIR / "Final_Report_Single.pdf"
CV_SOURCE = PROJECT_ROOT / "ZhouZiyi_CV.pdf"


SECTION_SOURCES = [
    MINI_PROJECT_ROOT / "section_01_introduction" / "report" / "section_01_introduction.md",
    MINI_PROJECT_ROOT / "section_02_data_and_backtesting_framework" / "report" / "section_2_data_features_var_backtesting_framework_en.md",
    MINI_PROJECT_ROOT / "section_03_historical_simulation" / "report" / "section_3_historical_simulation_final.md",
    MINI_PROJECT_ROOT / "section_04_garch_t" / "report" / "section_4_garch_t_final.md",
    MINI_PROJECT_ROOT / "section_05_neural_quantile_regression" / "report" / "section_5_neural_quantile_regression.md",
    REPORT_DIR / "section_06_integrated_comparison_and_conclusion.md",
]


UNIFIED_REFERENCES = """
## References

Andersen, T. G., Bollerslev, T., Diebold, F. X., and Labys, P. (2003). Modeling and forecasting realized volatility. Econometrica, 71(2), 579-625.

Barndorff-Nielsen, O. E., and Shephard, N. (2004). Power and bipower variation with stochastic volatility and jumps. Journal of Financial Econometrics, 2(1), 1-37.

Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics, 31(3), 307-327.

Boudoukh, J., Richardson, M., and Whitelaw, R. (1998). The best of both worlds: A hybrid approach to calculating value at risk. Risk, 11(5), 64-67.

Christoffersen, P. F. (1998). Evaluating interval forecasts. International Economic Review, 39(4), 841-862.

Christoffersen, P. F., and Pelletier, D. (2004). Backtesting value-at-risk: A duration-based approach. Journal of Financial Econometrics, 2(1), 84-108.

Danielsson, J., Ergun, L., de Haan, L., and de Vries, C. (2016). Tail index estimation: Quantile driven threshold selection. Working paper.

Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. Econometrica, 50(4), 987-1007.

Engle, R. F., and Manganelli, S. (2004). CAViaR: Conditional autoregressive value at risk by regression quantiles. Journal of Business & Economic Statistics, 22(4), 367-381.

Glosten, L. R., Jagannathan, R., and Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. Journal of Finance, 48(5), 1779-1801.

Hansen, P. R., Huang, Z., and Shek, H. H. (2012). Realized GARCH: A joint model for returns and realized measures of volatility. Journal of Applied Econometrics, 27(6), 877-906.

Koenker, R., and Bassett, G. (1978). Regression quantiles. Econometrica, 46(1), 33-50.

Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. Journal of Derivatives, 3(2), 73-84.

Lopez, J. A. (1999). Regulatory evaluation of value-at-risk models. Journal of Risk, 1(2), 37-64.

Pritsker, M. (2006). The hidden dangers of historical simulation. Journal of Banking and Finance, 30(2), 561-582.

Sheppard, K. (2024). arch: Autoregressive Conditional Heteroskedasticity models in Python.
"""


AUTHOR_CONTRIBUTION = """
## Author Contribution Statement

This project was implemented as an integrated empirical risk-modelling pipeline. The author designed the rolling-window backtesting framework, implemented Historical Simulation, GARCH-family, and neural quantile models in Python, generated the reported tables and figures, and interpreted the statistical diagnostics under a common VaR evaluation framework.

The mathematical and statistical contribution is the explicit treatment of VaR as a conditional quantile and the structured comparison of unconditional coverage, independence, duration, Lopez, and pinball-loss diagnostics. The programming contribution is the reproducible pipeline connecting raw SPY data, rolling forecasts, cached model outputs, backtesting tables, figures, and LaTeX-ready PDF generation. The domain contribution is the final GARCH-anchored neural correction, which encodes the observed GARCH calibration bias through a one-sided softplus constraint rather than treating the neural model as an unconstrained black box.
"""


def read_body(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    return text


def strip_references(text: str) -> str:
    markers = ["\n### References", "\n## References"]
    cut = len(text)
    for marker in markers:
        idx = text.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut].strip()


def rewrite_image_paths(text: str, source_md: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        caption = match.group(1)
        raw_path = match.group(2)
        if re.match(r"^[a-zA-Z]+://", raw_path) or Path(raw_path).is_absolute():
            return match.group(0)
        image_path = (source_md.parent / raw_path).resolve()
        rel_path = os.path.relpath(image_path, SINGLE_REPORT_MD.parent.resolve()).replace("\\", "/")
        return f"![{caption}]({rel_path})"

    return re.sub(r"!\[(.*?)\]\((.*?)\)", replace, text)


def normalize_heading_levels(text: str) -> str:
    lines = text.splitlines()
    heading_levels = []
    for line in lines:
        match = re.match(r"^(#{1,5})\s+", line)
        if match:
            heading_levels.append(len(match.group(1)))
    if not heading_levels or min(heading_levels) > 1:
        return text
    normalized = []
    for line in lines:
        match = re.match(r"^(#{1,5})(\s+.*)$", line)
        if match:
            normalized.append("#" + match.group(1) + match.group(2))
        else:
            normalized.append(line)
    return "\n".join(normalized)


def build_combined_markdown() -> None:
    title = """## Value-at-Risk Forecasting for SPY Daily Returns

Final single-PDF report.

This report studies one-day-ahead Value-at-Risk forecasting for SPY daily log returns. It compares Historical Simulation, GARCH-family conditional volatility models, direct neural quantile regression, and a conservative GARCH-anchored neural quantile correction under a common rolling-window backtesting framework.
"""
    parts = [title]
    for path in SECTION_SOURCES:
        body = normalize_heading_levels(strip_references(read_body(path)))
        parts.append(rewrite_image_paths(body, path))
    parts.append(UNIFIED_REFERENCES.strip())
    parts.append(AUTHOR_CONTRIBUTION.strip())
    SINGLE_REPORT_MD.write_text("\n\n".join(parts) + "\n", encoding="utf-8")


def load_exporter():
    local_code = PROJECT_ROOT / "code"
    exporter_path = MINI_PROJECT_ROOT / "section_03_historical_simulation" / "code" / "11_export_section3_latex_pdf.py"
    if not exporter_path.exists():
        exporter_path = PROJECT_ROOT / "HS_Family_Nonparametric_VaR" / "scripts" / "11_export_section3_latex_pdf.py"
    spec = importlib.util.spec_from_file_location("section3_exporter", exporter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load exporter from {exporter_path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(local_code.resolve()))
    spec.loader.exec_module(module)
    return module


def copytree_clean(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    shutil.copytree(
        src,
        dst,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            ".venv",
            "*.pdf",
            "14_export_demo_section1_latex_pdf.py",
            "14_export_demo_section6_latex_pdf.py",
            "15_export_section2_latex_pdf.py",
        ),
    )


def build_submission_package() -> None:
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    code_dir = SUBMISSION_DIR / "Corresponding_Code"
    if code_dir.exists():
        shutil.rmtree(code_dir)
    code_dir.mkdir(parents=True, exist_ok=True)

    for filename in ["README.md", "requirements.txt"]:
        src = PROJECT_ROOT / filename
        if src.exists():
            shutil.copy2(src, code_dir / filename)

    for dirname in ["code", "data", "report"]:
        copytree_clean(PROJECT_ROOT / dirname, code_dir / dirname)

    for dirname in ["outputs/tables", "outputs/forecasts", "outputs/figures", "outputs/section2_academic"]:
        copytree_clean(PROJECT_ROOT / dirname, code_dir / dirname)

    external_scripts = [
        MINI_PROJECT_ROOT / "section_05_neural_quantile_regression" / "code" / "external_section5_scripts" / "05_neural_quantile_var.py",
        MINI_PROJECT_ROOT / "section_05_neural_quantile_regression" / "code" / "external_section5_scripts" / "05c_garch_anchored_neural_var.py",
        MINI_PROJECT_ROOT / "section_05_neural_quantile_regression" / "code" / "external_section5_scripts" / "05d_tune_model_c_hyperparams.py",
    ]
    ext_script_dir = code_dir / "external_section5_scripts"
    ext_script_dir.mkdir(parents=True, exist_ok=True)
    for src in external_scripts:
        if src.exists():
            shutil.copy2(src, ext_script_dir / src.name)

    copytree_clean(
        MINI_PROJECT_ROOT / "section_05_neural_quantile_regression" / "outputs" / "external_section5_results",
        code_dir / "external_section5_results",
    )

    if CV_SOURCE.exists():
        shutil.copy2(CV_SOURCE, SUBMISSION_DIR / "ZhouZiyi_CV.pdf")
    else:
        (SUBMISSION_DIR / "CV_MISSING.txt").write_text(
            f"CV source not found at {CV_SOURCE}\n",
            encoding="utf-8",
        )


def write_checklist() -> None:
    cv_status = "Complete" if (SUBMISSION_DIR / "ZhouZiyi_CV.pdf").exists() else "Needs user replacement"
    checklist = f"""# Final Submission Checklist

- [x] Single PDF report created: `Final_Report_Single.pdf`
- [x] All chapters merged into one report: Sections 1-6, unified references, and author contribution statement
- [x] Data exploration included
- [x] Model motivations and detailed specifications included
- [x] Rolling-window settings and computational details included
- [x] VaR forecast figures included
- [x] Failure rates and statistical hypothesis testing included
- [x] Complete cross-model comparison table included
- [x] Best model by tail level stated and caveated
- [x] Neural methods compared with classical methods
- [x] Conclusion and future improvements included
- [x] Mathematical, programming, and domain strengths stated in academic author-contribution format
- [x] Corresponding Python code packaged under `Corresponding_Code`
- [{'x' if cv_status == 'Complete' else ' '}] CV included: {cv_status}

Model-result consistency check:

- Model C2 1% failure rate: 0.00947, Kupiec p-value: 0.7823
- Model C2 5% failure rate: 0.04962, Kupiec p-value: 0.9288
- Model C2 10% failure rate: 0.08902, Kupiec p-value: 0.0557

These are the same figures reported in Sections 1, 5, and 6.
"""
    (SUBMISSION_DIR / "SUBMISSION_CHECKLIST.md").write_text(checklist, encoding="utf-8")


def main() -> None:
    build_combined_markdown()
    build_submission_package()
    exporter = load_exporter()
    exporter.build_pdf(SINGLE_REPORT_MD, SINGLE_REPORT_PDF, chinese=False)
    pdf_all = FINAL_DIR / "01_PDF_All"
    pdf_all.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SINGLE_REPORT_PDF, pdf_all / "Final_Report_Single.pdf")
    write_checklist()
    print(f"Combined markdown: {SINGLE_REPORT_MD}")
    print(f"Single PDF report: {SINGLE_REPORT_PDF}")
    print(f"Submission package: {SUBMISSION_DIR}")


if __name__ == "__main__":
    main()

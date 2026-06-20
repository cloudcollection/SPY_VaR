from __future__ import annotations

import importlib.util
from pathlib import Path

from utils import PROJECT_ROOT


SECTION2_MD = PROJECT_ROOT / "report" / "section_2_data_visualization_empirical_analysis_zh.md"
SECTION2_PDF_SOURCE = PROJECT_ROOT / "report" / "section_2_data_visualization_pdf_source_zh.md"
SECTION2_PDF = PROJECT_ROOT / "pdf" / "section_02_data_visualization_zh.pdf"


FIGURES = [
    (
        "图 2-1 SPY日对数收益率及下尾观测值时间序列",
        "../outputs/section2/figures/fig2_1_log_returns_tail_events.png",
    ),
    (
        "图 2-2 已实现波动率与双幂变差的时间变化",
        "../outputs/section2/figures/fig2_2_realized_volatility_and_bv.png",
    ),
    (
        "图 2-3 SPY日对数收益率的经验分布",
        "../outputs/section2/figures/fig2_3_return_distribution_normal_overlay.png",
    ),
    (
        "图 2-4 SPY日对数收益率的正态Q-Q图",
        "../outputs/section2/figures/fig2_4_qq_plot_lower_tail.png",
    ),
    (
        "图 2-5 SPY日对数收益率的60日滚动波动率",
        "../outputs/section2/figures/fig2_5_rolling_volatility_regimes.png",
    ),
    (
        "图 2-6 收益率与平方收益率的自相关函数",
        "../outputs/section2/figures/fig2_6_acf_returns_and_squared_returns.png",
    ),
    (
        "图 2-7 收益率与波动率变量的相关系数矩阵",
        "../outputs/section2/figures/fig2_7_correlation_heatmap.png",
    ),
    (
        "图 2-8 已实现波动率与双幂变差的关系",
        "../outputs/section2/figures/fig2_8_rv5_bv_hexbin.png",
    ),
    (
        "图 2-9 按年份和月份划分的月度已实现波动率",
        "../outputs/section2/figures/fig2_9_monthly_volatility_heatmap.png",
    ),
]


def load_section3_exporter():
    exporter_path = Path(__file__).with_name("11_export_section3_latex_pdf.py")
    spec = importlib.util.spec_from_file_location("section3_exporter", exporter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load exporter from {exporter_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_pdf_source() -> Path:
    text = SECTION2_MD.read_text(encoding="utf-8")
    text = text.replace("# 第二章 数据特征分析与 VaR 回测框架", "## 第二章 数据特征分析与 VaR 回测框架", 1)

    marker = "\n## 2.6 图表清单\n"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip()

    figure_lines = ["", "## 2.6 学术图表输出", ""]
    for caption, rel_path in FIGURES:
        figure_lines.append(f"![{caption}]({rel_path})")
        figure_lines.append("")

    SECTION2_PDF_SOURCE.write_text(text + "\n" + "\n".join(figure_lines), encoding="utf-8")
    return SECTION2_PDF_SOURCE


def main() -> None:
    exporter = load_section3_exporter()
    source = build_pdf_source()
    exporter.build_pdf(source, SECTION2_PDF, chinese=True)


if __name__ == "__main__":
    main()

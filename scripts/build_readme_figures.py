"""Build factual, editable paper-style diagrams with EditableImage2PPTSkill.

All layouts/crops/QA reports go to --work-dir. Only the final PPTX is written to
--output-dir; render that saved deck to PNG with export_readme_figures.ps1.
Charts use the checked-in SPY data and archived forecast CSVs, never mock data.
External tool: soulmujoco/EditableImage2PPTSkill, commit 741a279e99da3b6818aa4c06f4ce8f151f1443ce.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spy_var.backtesting import evaluate
from spy_var.artifacts import archived_c2, load_archived_forecasts
from spy_var.config import MODEL_NAMES
from spy_var.data import load_data

BLUE, PURPLE, TEAL, RED = "#DCE7F6", "#E7DFF1", "#E1EEE7", "#F5DEDC"
INK, MUTED, EDGE = "#202124", "#555B63", "#66717E"
W, H = 1600, 900


def text(value, box, size=16, **extra):
    return dict(type="text", name=extra.pop("name", "label"), text=value, box=box,
                font="Times New Roman", size=size, color=INK, margin=1,
                align="center", valign="middle", **extra)


def shape(box, fill="none", line=EDGE, kind="rounded_rect", name="frame", **extra):
    return dict(type="shape", name=name, shape=kind, box=box, fill=fill,
                line=line, line_width=1, radius=0.05, **extra)


def arrow(x1, y1, x2, y2, color=EDGE, dash=None):
    return dict(type="line", name="arrow-flow", points=[x1, y1, x2, y2], line=color, line_width=1.4, dash=dash)


def rule(x1, y1, x2, y2, color=EDGE, dash=None):
    return dict(type="line", name="rule", points=[x1, y1, x2, y2], line=color, line_width=0.8, dash=dash)


def picture(name, box):
    return dict(type="image", name=name, path=f"assets/{name}.png", box=box)


def box_label(label, box, fill, size=16):
    return [shape(box, fill=fill), text(label, box, size)]


def sample_network(x, y, widths=(18, 64, 32, 3)):
    """Representative neurons; exact widths are stated, no weights are invented."""
    elements = []
    counts = [4, 5, 4, 3]
    columns = []
    for i, count in enumerate(counts):
        ys = np.linspace(y - 95, y + 95, count)
        columns.append([(x + i * 130, float(v)) for v in ys])
    for before, after in zip(columns, columns[1:]):
        for a in before:
            for b in after:
                elements.append(rule(*a, *b, color="#B4B9C1"))
    for i, points in enumerate(columns):
        for px, py in points:
            elements.append(shape([px - 9, py - 9, 18, 18], fill="#FFFFFF", kind="oval"))
        elements.append(text(str(widths[i]), [x + i * 130 - 35, y + 112, 70, 35], 16))
    return elements


def prepare_charts(work):
    data = load_data(ROOT / "code/data/spy_data.csv")
    frames, _ = load_archived_forecasts(ROOT, include_c2=True)
    c2 = archived_c2(ROOT)
    table, _ = evaluate(frames, calendar=data.index)
    table.to_csv(work / "historical_comparison.csv", index=False)
    paths = {}
    style = {"font.family": "DejaVu Serif", "font.size": 10, "axes.spines.top": False,
             "axes.spines.right": False, "axes.labelcolor": INK, "text.color": INK}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(5.4, 2.1), layout="constrained")
        ax.plot(data.index, 100 * data.log_ret, color="#506C91", linewidth=0.55)
        ax.set(ylabel="Daily log return (%)", xlabel="Date")
        fig.savefig(work / "returns.png", dpi=220, facecolor="white")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(3.5, 2.1), layout="constrained")
        ax.hist(100 * data.log_ret, bins=60, color="#CCD7E9", edgecolor="#53647B", linewidth=0.45)
        ax.set(xlabel="Daily log return (%)", ylabel="Count")
        ax.set_title(f"All {len(data):,} observations; 60 equal-width bins", fontsize=8)
        fig.savefig(work / "distribution.png", dpi=220, facecolor="white")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(12.5, 2.6), layout="constrained")
        dates = pd.to_datetime(c2["date/index"])
        ax.plot(dates, 100 * c2.realized_return, color="#B6BCC4", linewidth=0.6, label="Observed return")
        ax.plot(dates, 100 * c2["VaR_0.01"], color="#385F87", linewidth=0.85, label="Archived C2 1% quantile")
        missed = c2.realized_return < c2["VaR_0.01"]
        ax.scatter(dates[missed], 100 * c2.loc[missed, "realized_return"], s=14, color="#B8504B", marker="x", label="Violations", zorder=4)
        ax.set(ylabel="Daily log return (%)", xlabel="Target date")
        ax.legend(loc="upper right", frameon=False, ncol=3, fontsize=9)
        fig.savefig(work / "c2_trace.png", dpi=220, facecolor="white")
        plt.close(fig)
        labels = ["Time-weighted HS", "KDE weighted HS", "GARCH-t", "GJR-GARCH-t", "Lag-window MLP", "C2 correction"]
        fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.35), layout="constrained")
        for ax, alpha in zip(axes, [0.01, 0.05, 0.1]):
            subset = table.loc[table.alpha == alpha]
            colors = ["#8C9BAF"] * 5 + ["#456F94"]
            ax.barh(labels, subset.failure_rate.to_numpy() * 100, color=colors, height=0.6)
            ax.axvline(alpha * 100, color="#A84D49", linestyle="--", linewidth=1.2)
            ax.set(title=f"{alpha:.0%} quantile", xlabel="Observed violation rate (%)")
            ax.set_xlim(left=0)
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.15)
        fig.savefig(work / "calibration.png", dpi=220, facecolor="white")
        plt.close(fig)
        # Exact observed return values from the first new-model input window.
        fig, ax = plt.subplots(figsize=(3.8, 1.65), layout="constrained")
        ax.plot(np.arange(64), 100 * data.log_ret.iloc[936:1000], color="#506C91", linewidth=1.0)
        ax.set(xlabel="Observed lag position", ylabel="Return (%)")
        fig.savefig(work / "patch_input.png", dpi=220, facecolor="white")
        plt.close(fig)
    for name in ["returns", "distribution", "c2_trace", "calibration", "patch_input"]:
        paths[name] = work / f"{name}.png"
    # Build a source image containing only our data plots; crop the actual panels.
    # No plot is erased, enhanced or borrowed from the reference screenshots.
    sheet_width = max(Image.open(path).width for path in paths.values())
    sheet_height = sum(Image.open(path).height for path in paths.values())
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    assets, y = [], 0
    for name, path in paths.items():
        image = Image.open(path).convert("RGB")
        sheet.paste(image, (0, y))
        assets.append({"name": name + ".png", "box": [0, y, image.width, y + image.height], "mode": "keep"})
        y += image.height
    sheet.save(work / "source_data_panels.png")
    (work / "assets.json").write_text(json.dumps({"source": "source_data_panels.png", "assets": assets}), encoding="utf-8")
    audit = {"data_sha256": sha256((ROOT / "code/data/spy_data.csv").read_bytes()).hexdigest(),
             "n": len(data), "min_return": float(data.log_ret.min()), "max_return": float(data.log_ret.max()),
             "histogram_outside_display": 0,
             "n_c2_forecasts": len(c2), "n_c2_1pct_violations": int(missed.sum())}
    (work / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return table


def workflow():
    e = [text("One-day SPY quantile forecasting", [35, 10, 1530, 60], 25, bold=True),
         text("(a) Observed data", [40, 82, 440, 42], 18), picture("returns", [40, 128, 520, 208]),
         text("(b) Empirical return distribution", [590, 82, 390, 42], 18), picture("distribution", [590, 128, 350, 208]),
         text("(c) Chronological estimation", [995, 82, 560, 42], 18),
         text("Forecast target: day t", [1010, 135, 520, 35], 16)]
    e += box_label("Training", [1020, 200, 290, 75], BLUE)
    e += box_label("Validation", [1310, 200, 140, 75], PURPLE, 14)
    e += box_label("t", [1467, 200, 55, 75], RED)
    e += [arrow(1015, 303, 1535, 303), text("Past observations only; roll forward one day", [995, 315, 560, 45], 14),
          rule(35, 370, 1565, 370, dash="dash"),
          text("(d) Implemented model families and shared evaluation", [40, 382, 1520, 48], 19)]
    e += box_label("SPY snapshot\n4,640 trading observations\nlog_ret  ·  rv5  ·  bv", [45, 525, 280, 145], BLUE, 16)
    e += [arrow(325, 598, 385, 598), rule(385, 483, 385, 737), arrow(385, 483, 450, 483), arrow(385, 737, 450, 737)]
    e += box_label("Statistical baselines\nOrdinary / weighted / KDE HS\nGARCH-t  ·  GJR-GARCH-t", [450, 440, 445, 125], TEAL, 16)
    e += box_label("Neural quantile models\nMLP  ·  C2 anchored correction\nPatchTST-inspired Transformer", [450, 665, 445, 125], PURPLE, 16)
    e += [arrow(670, 565, 670, 665, color="#8E6B9E"), text("GARCH anchor → C2", [700, 595, 260, 35], 14),
          arrow(895, 503, 985, 503), arrow(895, 727, 985, 727), rule(985, 503, 985, 727), arrow(985, 612, 1040, 612)]
    e += box_label("Three ordered quantiles\nq₀.₀₁,t ≤ q₀.₀₅,t ≤ q₀.₁₀,t", [1040, 550, 270, 115], BLUE, 17)
    e += [arrow(1310, 608, 1360, 608)]
    e += box_label("Shared target dates\nCoverage /\nindependence\nPinball loss\nFit diagnostics", [1360, 510, 205, 190], "#F7F4EB", 13)
    e += [text("Data: 2000-01-04–2018-06-27  ·  horizon: one trading day  ·  α ∈ {0.01, 0.05, 0.10}", [35, 828, 1530, 36], 14)]
    return e


def anchored_architecture():
    e = [text("Conservative GARCH-anchored neural correction (C2)", [30, 10, 1540, 60], 25, bold=True),
         shape([35, 105, 350, 440], fill="none", name="dashed-feature-group"),
         text("(a) Observed feature vector", [45, 112, 330, 42], 18)]
    e += box_label("Lagged returns and\nrolling statistics\n10 features", [65, 180, 290, 95], BLUE, 16)
    e += box_label("Lagged rv5 and bv\n4 features", [65, 300, 290, 80], TEAL, 16)
    e += box_label("GARCH quantiles + σ\n4 features", [65, 405, 290, 85], RED, 16)
    e += [text("18 inputs; training-only standardization", [35, 555, 370, 55], 14), arrow(385, 320, 465, 320),
          text("(b) Dense correction network", [450, 100, 555, 45], 18)]
    e += sample_network(500, 320)
    e += [text("18 → 64 → 32 → 3", [475, 500, 455, 45], 18),
          text("ReLU · dropout 0.1 after first hidden layer\nCircles represent dense layers; widths are exact", [435, 555, 565, 65], 14),
          arrow(920, 320, 1040, 320)]
    e += box_label("Raw correction gθ(x)\n3 outputs", [1040, 275, 240, 85], PURPLE, 16)
    e += [arrow(1160, 360, 1160, 427)]
    e += box_label("s · softplus(gθ(x))\ns = 1; nonnegative", [1040, 430, 240, 85], PURPLE, 16)
    e += [arrow(1280, 474, 1360, 474)]
    e += box_label("Subtract from\nGARCH quantiles", [1360, 430, 195, 85], BLUE, 16)
    e += [rule(350, 445, 410, 445, color="#A45F5A"), rule(410, 445, 410, 190, color="#A45F5A"),
          arrow(410, 190, 1460, 190, color="#A45F5A"), arrow(1460, 190, 1460, 430, color="#A45F5A"),
          text("One-step GARCH-t anchor", [1040, 135, 505, 40], 17),
          arrow(1460, 515, 1460, 565)]
    e += box_label("Sort across α\nFinal return quantiles", [1335, 570, 245, 85], TEAL, 16)
    e += [shape([50, 700, 1490, 130], fill="#FAFAFA", line="#959DA8"),
          text("q̂C2α,t = sortα [q̂GARCHα,t − s · softplus(gθ(x))α]", [85, 715, 1420, 48], 23, name="formula-c2"),
          text("Training: mean pinball loss over α = 0.01, 0.05, 0.10; chronological early stopping", [85, 773, 1420, 35], 15),
          text("Implemented project model; inherited regularization was selected after inspecting historical test results", [35, 853, 1530, 30], 13)]
    return e


def patch_architecture():
    e = [text("Patch-based Transformer for one-day return quantiles", [30, 10, 1540, 60], 25, bold=True),
         text("Project-specific adaptation inspired by PatchTST (Nie et al., ICLR 2023)", [40, 77, 1520, 35], 17),
         shape([35, 145, 360, 300], fill="none", name="dashed-observed-window"),
         text("(a) Input: 64 observed log returns", [45, 150, 340, 45], 17),
         picture("patch_input", [50, 205, 325, 150]),
         text("One return channel; no future data", [50, 375, 325, 40], 14), arrow(395, 285, 455, 285)]
    e += box_label("Reversible per-window\nnormalization\n(no learnable affine)", [455, 220, 240, 130], BLUE, 16)
    e += [arrow(695, 285, 755, 285)]
    e += box_label("Patch tokenization\nlength 16 · stride 8\n7 overlapping patches", [755, 220, 260, 130], TEAL, 16)
    e += [arrow(1015, 285, 1070, 285)]
    e += box_label("Linear projection\n+ learned position\n7 tokens · width 32", [1070, 220, 235, 130], PURPLE, 16)
    e += [arrow(1305, 285, 1360, 285)]
    e += box_label("Transformer encoder\n2 layers · 4 heads\nFFN 64 · dropout 0.1", [1360, 210, 215, 150], BLUE, 15)
    e += [rule(1467, 360, 1467, 480), arrow(1467, 480, 1240, 480)]
    e += box_label("Flatten 7 × 32\nLinear 224 → 3", [1000, 430, 240, 100], PURPLE, 17)
    e += [arrow(1000, 480, 930, 480)]
    e += box_label("Inverse normalization\n+ quantile sorting", [665, 430, 265, 100], TEAL, 17)
    e += [arrow(665, 480, 635, 480)]
    e += box_label("q₀.₀₁,t, q₀.₀₅,t, q₀.₁₀,t", [410, 435, 225, 90], BLUE, 18)
    e += [text("(b) Native patch representation", [45, 485, 260, 45], 17)]
    for i in range(7):
        e.append(shape([55 + i * 30, 560 + i * 4, 150, 25], fill=[BLUE, PURPLE, TEAL][i % 3], kind="rect"))
    e += [text("Patch overlap is structural;\nno fabricated activations shown", [40, 650, 380, 60], 14),
          shape([460, 610, 1115, 160], fill="#FAFAFA", line="#959DA8"),
          text("(c) Quantile objective and chronological validation", [475, 620, 1080, 40], 18),
          text("L(θ) = meanα,t ρα(rt − q̂α,t);    ρα(u) = max(αu, (α − 1)u)", [480, 670, 1065, 45], 21, name="formula-pinball"),
          text("20-epoch cap · Adam 0.001 · batch 64 · 20% trailing validation · refit every 20 days", [480, 723, 1065, 35], 14),
          text("Univariate quantile adaptation; distinct from the original long-horizon point-forecasting model", [35, 830, 1530, 35], 15)]
    return e


def empirical():
    return [text("Archived forecasts on common target dates", [30, 5, 1540, 60], 25, bold=True),
            text("(a) Observed violation rates; dashed lines mark nominal probabilities", [40, 75, 1520, 40], 18),
            picture("calibration", [40, 125, 1520, 375]),
            text("(b) C2 1% quantile and all observed returns on the same 2,640 dates", [40, 512, 1520, 40], 18),
            picture("c2_trace", [70, 560, 1460, 285]),
            text("2008-01-03–2018-06-27 · exact archived CSVs · exploratory evidence; no superiority claim", [35, 855, 1530, 30], 13)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/figures")
    parser.add_argument("--ppt-python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    work = args.work_dir.resolve()
    if work == ROOT or ROOT in work.parents:
        raise ValueError("Use a private work directory outside the repository")
    work.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepare_charts(work)
    scripts = args.tool_root / "scripts"
    subprocess.run([str(args.ppt_python), "-B", str(scripts / "asset_cropper.py"), "--manifest", str(work / "assets.json"),
                    "--out-dir", str(work / "assets"), "--contact-sheet"], check=True)
    layouts = work / "layouts"
    layouts.mkdir(exist_ok=True)
    for i, elements in enumerate([workflow(), anchored_architecture(), patch_architecture(), empirical()], 1):
        for element in elements:
            if element["type"] == "image":
                image = Image.open(work / element["path"])
                x, y, w, h = element["box"]
                factor = min(w / image.width, h / image.height)
                new_w, new_h = image.width * factor, image.height * factor
                element["box"] = [x + (w - new_w) / 2, y + (h - new_h) / 2, new_w, new_h]
        doc = {"source_width": W, "source_height": H, "slide_size": {"width_in": 16, "height_in": 9},
               "background": "#FFFFFF", "elements": elements}
        (layouts / f"page_{i:03d}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    subprocess.run([str(args.ppt_python), "-B", str(scripts / "combine_layouts.py"), "--layouts", str(layouts),
                    "--out", str(work / "combined.json")], check=True)
    pptx = args.output_dir / "spy_var_methods.pptx"
    subprocess.run([str(args.ppt_python), "-B", str(scripts / "build_pptx_from_layout.py"), "--layout", str(work / "combined.json"),
                    "--assets-root", str(work), "--out", str(pptx)], check=True)
    subprocess.run([str(args.ppt_python), "-B", str(ROOT / "scripts/finish_editable_figures.py"), str(pptx)], check=True)
    subprocess.run([str(args.ppt_python), "-B", str(scripts / "inspect_pptx.py"), "--pptx", str(pptx),
                    "--report", str(work / "pptx_qa.json")], check=True)
    print(f"Final editable figures: {pptx}")


if __name__ == "__main__":
    main()

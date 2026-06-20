from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from utils import PROJECT_ROOT


EN_MD = PROJECT_ROOT / "report" / "section_3_historical_simulation_final.md"
ZH_MD = PROJECT_ROOT / "report" / "section_3_historical_simulation_final_zh.md"
BUILD_DIR = PROJECT_ROOT / "outputs" / "latex_build"
DOWNLOADS = Path.home() / "Downloads"
XELATEX_FALLBACK = None


def find_xelatex() -> str:
    found = shutil.which("xelatex")
    if found:
        return found
    if XELATEX_FALLBACK and XELATEX_FALLBACK.exists():
        return str(XELATEX_FALLBACK)
    raise FileNotFoundError("xelatex was not found on PATH. Install TeX Live or add xelatex to PATH.")


def escape_latex(text: str) -> str:
    text = text.replace("\\", r"\textbackslash{}")
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def render_inline(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)

    code_spans: list[str] = []

    def keep_code(match: re.Match[str]) -> str:
        code = escape_latex(match.group(1))
        code_spans.append(rf"\texttt{{{code}}}")
        return f"@@CODE{len(code_spans)-1}@@"

    text = re.sub(r"`([^`]+)`", keep_code, text)

    math_spans: list[str] = []

    def keep_math(match: re.Match[str]) -> str:
        math_spans.append(match.group(0))
        return f"@@MATH{len(math_spans)-1}@@"

    # Protect formulas that the Markdown source already marks correctly.
    text = re.sub(r"\$[^$]+\$", keep_math, text)

    # Normalize remaining common inline mathematical expressions before
    # escaping prose. Display equations are handled separately.
    text = text.replace(r"Q_{\alpha}(\cdot)", r"$Q_{\alpha}(\cdot)$")
    text = text.replace(r"H_0:\pi_{01}=\pi_{11}", r"$H_0:\pi_{01}=\pi_{11}$")
    text = re.sub(r"\bH_0:p=alpha\b", lambda _m: r"$H_0:p=\alpha$", text)
    text = re.sub(r"\bp=alpha\b", lambda _m: r"$p=\alpha$", text)
    text = re.sub(r"\bV <= ([0-9]+)", r"$V \\le \1$", text)
    text = re.sub(r"\bV >= ([0-9]+)", r"$V \\ge \1$", text)
    text = re.sub(r"\bT = ([0-9]+)", r"$T=\1$", text)
    text = re.sub(r"\bW = ([0-9]+)", r"$W=\1$", text)
    text = re.sub(r"\balpha = ([0-9]+)\\?%", lambda m: rf"$\alpha={m.group(1)}\%$", text)
    text = re.sub(r"\blambda = ([0-9.]+)", lambda m: rf"$\lambda={m.group(1)}$", text)
    text = re.sub(r"\b49/3640 = 1\.35\\?%", lambda _m: r"$49/3640=1.35\%$", text)
    text = text.replace("n_eff", r"$n_{\mathrm{eff}}$")
    text = text.replace("sigma_w", r"$\sigma_w$")
    text = re.sub(r"(?<![A-Za-z])n\^\{-?[0-9]+/[0-9]+\}", lambda m: f"${m.group(0)}$", text)
    text = re.sub(r"\bT_\{[ij01]+\}", lambda m: f"${m.group(0)}$", text)
    text = re.sub(r"\btau_i\b", lambda _m: r"$\tau_i$", text)
    text = re.sub(r"\bpi_\{[01]+\}", lambda m: f"$\\{m.group(0)}$", text)
    text = re.sub(r"\blambda\b", lambda _m: r"$\lambda$", text)
    text = re.sub(r"\balpha\b", lambda _m: r"$\alpha$", text)

    # Protect formulas introduced by the normalization rules above.
    text = re.sub(r"\$[^$]+\$", keep_math, text)
    text = escape_latex(text)
    for idx, math in enumerate(math_spans):
        text = text.replace(f"@@MATH{idx}@@", math)
    for idx, code in enumerate(code_spans):
        text = text.replace(f"@@CODE{idx}@@", code)
    return text


def parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    table_lines = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        table_lines.append(lines[i].strip())
        i += 1
    rows = []
    for line in table_lines:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        rows.append(cells)
    return rows, i


def table_to_latex(rows: list[list[str]]) -> str:
    n_cols = len(rows[0])
    colspec = "@{}" + "l" + "".join("r" if idx > 0 else "l" for idx in range(1, n_cols)) + "@{}"
    out = [r"\begin{table}[H]", r"\centering", r"\scriptsize", r"\begin{adjustbox}{max width=\textwidth}", rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"]
    header = " & ".join(render_inline(cell) for cell in rows[0]) + r" \\"
    out.append(header)
    out.append(r"\midrule")
    for row in rows[1:]:
        padded = row + [""] * (n_cols - len(row))
        out.append(" & ".join(render_inline(cell) for cell in padded[:n_cols]) + r" \\")
    out.extend([r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"])
    return "\n".join(out)


def image_to_latex(caption: str, rel_path: str, md_path: Path) -> str:
    image_path = (md_path.parent / rel_path).resolve()
    return "\n".join(
        [
            r"\begin{figure}[H]",
            r"\centering",
            rf"\includegraphics[width=0.92\textwidth]{{{image_path.as_posix()}}}",
            rf"\caption*{{{render_inline(caption)}}}",
            r"\end{figure}",
        ]
    )


def markdown_to_latex(md_path: Path) -> str:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    paragraph: list[str] = []
    i = 0

    def flush_paragraph() -> None:
        if paragraph:
            text = " ".join(part.strip() for part in paragraph if part.strip())
            if text:
                out.append(render_inline(text) + "\n")
            paragraph.clear()

    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            i += 1
            continue

        if stripped.startswith("$$"):
            flush_paragraph()
            formula_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("$$"):
                formula_lines.append(lines[i].strip())
                i += 1
            formula = "\n".join(formula_lines).strip()
            tag_match = re.search(r"\\tag\{([^}]+)\}", formula)
            formula = re.sub(r"\\tag\{[^}]+\}", "", formula).strip()
            if tag_match:
                out.append(r"\begin{equation}" + "\n" + formula + rf"\tag{{{tag_match.group(1)}}}" + "\n" + r"\end{equation}")
            else:
                out.append(r"\[" + "\n" + formula + "\n" + r"\]")
            i += 1
            continue

        image_match = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
        if image_match:
            flush_paragraph()
            out.append(image_to_latex(image_match.group(1), image_match.group(2), md_path))
            i += 1
            continue

        if stripped.startswith("|"):
            flush_paragraph()
            rows, next_i = parse_table(lines, i)
            if rows:
                out.append(table_to_latex(rows))
            i = next_i
            continue

        if stripped.startswith("## "):
            flush_paragraph()
            out.append(rf"\section*{{{render_inline(stripped[3:])}}}")
        elif stripped.startswith("### "):
            flush_paragraph()
            out.append(rf"\subsection*{{{render_inline(stripped[4:])}}}")
        elif stripped.startswith("#### "):
            flush_paragraph()
            out.append(rf"\subsubsection*{{{render_inline(stripped[5:])}}}")
        elif stripped.startswith("- "):
            flush_paragraph()
            out.append(rf"\noindent -- {render_inline(stripped[2:])}")
        else:
            paragraph.append(stripped)
        i += 1

    flush_paragraph()
    return "\n\n".join(out)


def preamble(chinese: bool) -> str:
    cjk = "\n".join(
        [
            r"\usepackage{xeCJK}",
            r"\setCJKmainfont{SimSun}",
        ]
    ) if chinese else ""
    return rf"""
\documentclass[11pt,a4paper]{{article}}
\usepackage[margin=0.75in]{{geometry}}
\usepackage{{fontspec}}
{cjk}
\setmainfont{{Times New Roman}}
\usepackage{{unicode-math}}
\IfFontExistsTF{{TeX Gyre Termes Math}}{{\setmathfont{{TeX Gyre Termes Math}}}}{{}}
\usepackage{{amsmath}}
\usepackage{{booktabs,array,longtable,adjustbox,float}}
\usepackage{{graphicx}}
\usepackage{{caption}}
\usepackage{{hyperref}}
\hypersetup{{colorlinks=false,pdfborder={{0 0 0}}}}
\setlength{{\parindent}}{{1.8em}}
\setlength{{\parskip}}{{0.35em}}
\linespread{{1.08}}
\captionsetup{{font=small,labelfont=bf}}
\begin{{document}}
"""


def build_pdf(md_path: Path, output_pdf: Path, chinese: bool) -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    stem = output_pdf.stem
    tex_path = BUILD_DIR / f"{stem}.tex"
    body = markdown_to_latex(md_path)
    tex_path.write_text(preamble(chinese) + body + "\n\\end{document}\n", encoding="utf-8")
    xelatex = find_xelatex()
    for _ in range(2):
        subprocess.run(
            [xelatex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=BUILD_DIR,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    compiled = BUILD_DIR / f"{stem}.pdf"
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(compiled, output_pdf)
    print(f"Exported native-LaTeX PDF to {output_pdf}")


def main() -> None:
    build_pdf(EN_MD, DOWNLOADS / "section_3_historical_simulation_latex_audited_english.pdf", chinese=False)
    build_pdf(ZH_MD, DOWNLOADS / "section_3_historical_simulation_latex_audited_chinese.pdf", chinese=True)


if __name__ == "__main__":
    main()

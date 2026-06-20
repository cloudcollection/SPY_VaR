from __future__ import annotations

import importlib.util
from pathlib import Path

from utils import PROJECT_ROOT


SECTION6_MD = PROJECT_ROOT / "report" / "Demo_Section6.md"
SECTION6_PDF = PROJECT_ROOT / "outputs" / "pdf" / "Demo_Section6.pdf"


def load_section3_exporter():
    exporter_path = PROJECT_ROOT / "HS_Family_Nonparametric_VaR" / "scripts" / "11_export_section3_latex_pdf.py"
    spec = importlib.util.spec_from_file_location("section3_exporter", exporter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load exporter from {exporter_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    exporter = load_section3_exporter()
    exporter.build_pdf(SECTION6_MD, SECTION6_PDF, chinese=False)


if __name__ == "__main__":
    main()

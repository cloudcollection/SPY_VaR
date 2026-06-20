from __future__ import annotations

import importlib.util
from pathlib import Path

from utils import PROJECT_ROOT


SECTION5_MD = PROJECT_ROOT / "report" / "section_5_neural_quantile_regression.md"
SECTION5_PDF = PROJECT_ROOT / "pdf" / "section_05_neural_quantile_regression.pdf"


def load_section3_exporter():
    exporter_path = Path(__file__).with_name("11_export_section3_latex_pdf.py")
    spec = importlib.util.spec_from_file_location("section3_exporter", exporter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load exporter from {exporter_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    SECTION5_PDF.parent.mkdir(parents=True, exist_ok=True)
    exporter = load_section3_exporter()
    exporter.build_pdf(SECTION5_MD, SECTION5_PDF, chinese=False)


if __name__ == "__main__":
    main()

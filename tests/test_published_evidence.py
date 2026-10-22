from hashlib import sha256
from pathlib import Path
import re
from xml.etree import ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

from spy_var.artifacts import load_archived_forecasts
from spy_var.backtesting import evaluate
from spy_var.data import load_data

ROOT = Path(__file__).resolve().parents[1]


def test_published_historical_table_is_recomputed_from_actual_predictions():
    frames, _ = load_archived_forecasts(ROOT, include_c2=True)
    calculated, _ = evaluate(frames, calendar=load_data(ROOT / "code/data/spy_data.csv").index)
    published = pd.read_csv(ROOT / "docs/results/historical_comparison.csv")
    pd.testing.assert_frame_equal(calculated, published, check_exact=False, atol=1e-12, rtol=1e-10)
    assert published.n_obs.eq(2640).all()


def test_readme_tables_are_generated_from_final_numeric_evidence():
    import importlib.util
    spec = importlib.util.spec_from_file_location("publication", ROOT / "scripts/publish_readme_results.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    historical = pd.read_csv(ROOT / "docs/results/historical_comparison.csv")
    modern = pd.read_csv(ROOT / "docs/results/patch_transformer_summary.csv")
    assert helper.historical_markdown(historical) in readme
    assert helper.modern_markdown(modern) in readme
    by_seed = pd.read_csv(ROOT / "docs/results/patch_transformer_by_seed.csv")
    assert set(by_seed.seed) == {42, 43, 44}
    assert by_seed.n_obs.eq(3640).all()
    assert np.allclose(modern.failure_rate_mean, by_seed.groupby("alpha").failure_rate.mean())
    assert np.allclose(modern.failure_rate_std, by_seed.groupby("alpha").failure_rate.std())


def test_result_evidence_identifies_the_actual_trained_source():
    digest = sha256((ROOT / "src/spy_var/models/patch_transformer.py").read_bytes()).hexdigest()
    assert digest in (ROOT / "docs/EXPERIMENTS.md").read_text(encoding="utf-8")


def test_editable_pptx_has_native_text_and_no_duplicate_arrow_or_effect_nodes():
    folder = ROOT / "docs/figures"
    assert {p.name for p in folder.iterdir()} == {
        "research_workflow.png", "c2_architecture.png", "patch_transformer.png",
        "historical_evaluation.png", "spy_var_methods.pptx",
    }
    ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    text_count = 0
    with zipfile.ZipFile(folder / "spy_var_methods.pptx") as package:
        slides = [name for name in package.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)]
        assert len(slides) == 4
        for slide in slides:
            root = ET.fromstring(package.read(slide))
            text_count += len(root.findall(".//a:t", ns))
            for node in root.findall(".//a:ln", ns):
                assert len(node.findall("a:tailEnd", ns)) <= 1
            for node in root.iter():
                assert len(node.findall("a:effectLst", ns)) <= 1
        assert text_count >= 90


def test_readme_uses_only_final_local_figure_files():
    content = (ROOT / "README.md").read_text(encoding="utf-8")
    for path in re.findall(r'<img src="([^"]+)"', content):
        assert (ROOT / path).is_file()
    for path in [ROOT / "README.md", *list((ROOT / "docs").glob("*.md"))]:
        text = path.read_text(encoding="utf-8")
        assert "C:/Users/" not in text and "C:\\Users\\" not in text

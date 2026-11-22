"""ALTUN 1 modelini Hugging Face'e yükler.
Kullanım: HF_TOKEN=... python scripts/publish_to_hf.py --results-dir <ALTUN-1-RESULTS> --repo-id erlic/ALTUN-1
Not: bu betik ağ gerektirir; geliştirme ortamında çalıştırılarak test EDİLMEDİ (yükleme sırasında hatalar net görünür)."""
import argparse
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, help="Açılmış ALTUN-1-RESULTS klasörü")
    ap.add_argument("--repo-id", default="erlic/ALTUN-1")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    fm = os.path.join(a.results_dir, "final_model")
    need = [os.path.join(fm, f) for f in ("model.safetensors", "config.json")] + [os.path.join(fm, "tokenizer")]
    missing = [p for p in need if not os.path.exists(p)]
    if missing:
        sys.exit(f"Eksik dosyalar: {missing}")
    from huggingface_hub import HfApi
    stage = tempfile.mkdtemp(prefix="altun_hf_")
    shutil.copy2(os.path.join(fm, "model.safetensors"), stage)
    shutil.copy2(os.path.join(fm, "config.json"), stage)
    shutil.copytree(os.path.join(fm, "tokenizer"), os.path.join(stage, "tokenizer"))
    shutil.copy2(os.path.join(ROOT, "MODEL_CARD.md"), os.path.join(stage, "README.md"))
    for f in ("LICENSE", "NOTICE"):
        shutil.copy2(os.path.join(ROOT, f), stage)
    os.makedirs(os.path.join(stage, "results"))
    shutil.copy2(os.path.join(ROOT, "results", "evaluation_summary.md"), os.path.join(stage, "results"))
    for sub, names in (("evaluations", ("custom_math_results.json", "failed_examples.json", "turkish_results.json", "benchmark_results.json", "test_results.json")),
                       ("reports", ("training_report.md", "training_result.json", "test_report.md", "provenance.json", "data_quality_report.md"))):
        for n in names:
            p = os.path.join(a.results_dir, sub, n)
            if os.path.isfile(p):
                shutil.copy2(p, os.path.join(stage, "results", n))
    api = HfApi(token=token)
    api.create_repo(a.repo_id, repo_type="model", private=a.private, exist_ok=True)
    api.upload_folder(folder_path=stage, repo_id=a.repo_id, repo_type="model", commit_message="ALTUN 1 v0.1")
    print(f"Yüklendi: https://huggingface.co/{a.repo_id}")


if __name__ == "__main__":
    main()

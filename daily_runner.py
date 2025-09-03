# daily_runner.py
# Runs the daily pipeline using cloud resources (HF model + Postgres),
# never relying on local model or local SQLite.

import os
import sys
import subprocess
from pathlib import Path

# ---- 1) Load secrets from .streamlit/secrets.toml ----
def load_secrets():
    """
    Load .streamlit/secrets.toml (same file Streamlit Cloud uses) and
    push all keys into os.environ so child processes see them.
    """
    secrets_path = Path(".streamlit/secrets.toml")
    if not secrets_path.exists():
        return {}

    try:
        import tomllib  # py3.11+
    except ModuleNotFoundError:
        import toml as tomllib  # fallback for older envs

    with secrets_path.open("rb") as f:
        data = tomllib.load(f)

    # Ensure all values are strings for env
    for k, v in data.items():
        os.environ[k] = str(v)
    return data


# ---- 2) Small helpers ----
def mask(val: str | None, keep: int = 3):
    if not val:
        return "(none)"
    if len(val) <= keep * 2:
        return val[0:1] + "…" + val[-1:]
    return f"{val[:keep]}…{val[-keep:]}"


def require_env(keys: list[str]):
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )


def run_module(mod: str, env: dict[str, str]):
    print(f"\nRunning {mod}...")
    # Use `python -m package.module` so imports resolve from project root
    subprocess.run([sys.executable, "-m", mod], check=True, env=env)


def main():
    # Load secrets first
    secrets = load_secrets()

    # 3) Force pipeline to use the HF-hosted model (not local)
    #    If you want a different repo, set it in secrets or override here.
    if not os.getenv("SENTIMENT_MODEL_REPO"):
        # Default to your private repo on HF
        os.environ["SENTIMENT_MODEL_REPO"] = "pranava145/my_sentiment_model"

    # Print masked env summary for sanity
    print(
        "ENV → DATABASE_URL:",
        mask(os.getenv("DATABASE_URL")),
        "| OPENAI_API_KEY:",
        mask(os.getenv("OPENAI_API_KEY")),
        "| HF_TOKEN:",
        mask(os.getenv("HF_TOKEN")),
        "| SENTIMENT_MODEL_REPO:",
        os.getenv("SENTIMENT_MODEL_REPO") or "(none)",
    )

    # Require the things we truly need for cloud resources
    require_env(["DATABASE_URL"])
    # score_sentiment uses HF repo + token
    require_env(["HF_TOKEN", "SENTIMENT_MODEL_REPO"])

    # Propagate current env to children
    child_env = os.environ.copy()

    # ---- 4) Run the daily jobs in order ----
    #     - scripts.collect_to_db       -> fetch headlines + upsert to Postgres
    #     - scripts.fill_missing_text   -> fetch article bodies for empty text
    #     - scripts.score_sentiment     -> loads model FROM HF and writes scores
    #     - scripts.build_embeddings    -> rebuild embeddings table
    try:
        run_module("scripts.collect_to_db", child_env)
        run_module("scripts.fill_missing_text", child_env)
        run_module("scripts.score_sentiment", child_env)   # uses HF model
        run_module("scripts.build_embeddings", child_env)
    except subprocess.CalledProcessError as e:
        # Surface the failing step cleanly
        print(f"\n Pipeline step failed: {e.args}")
        sys.exit(e.returncode)

    print("\n Daily pipeline complete (HF model + Postgres).")


if __name__ == "__main__":
    main()

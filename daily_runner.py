

import os
import sys
import subprocess
from pathlib import Path

def load_secrets():
    """
    Load .streamlit/secrets.toml (same file Streamlit Cloud uses) and
    push all keys into os.environ so child processes see them.
    """
    secrets_path = Path(".streamlit/secrets.toml")
    if not secrets_path.exists():
        return {}

    try:
        import tomllib  
    except ModuleNotFoundError:
        import toml as tomllib  

    with secrets_path.open("rb") as f:
        data = tomllib.load(f)

    for k, v in data.items():
        os.environ[k] = str(v)
    return data


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
    subprocess.run([sys.executable, "-m", mod], check=True, env=env)


def main():
    secrets = load_secrets()

    if not os.getenv("SENTIMENT_MODEL_REPO"):
        os.environ["SENTIMENT_MODEL_REPO"] = "pranava145/my_sentiment_model"

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

    require_env(["DATABASE_URL"])
    require_env(["HF_TOKEN", "SENTIMENT_MODEL_REPO"])

    child_env = os.environ.copy()


    try:
        run_module("scripts.collect_to_db", child_env)
        run_module("scripts.fill_missing_text", child_env)
        run_module("scripts.score_sentiment", child_env)   # uses HF model
        run_module("scripts.build_embeddings", child_env)
    except subprocess.CalledProcessError as e:
        print(f"\n Pipeline step failed: {e.args}")
        sys.exit(e.returncode)

    print("\n Daily pipeline complete (HF model + Postgres).")


if __name__ == "__main__":
    main()

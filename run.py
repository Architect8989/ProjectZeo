# run.py

import os
import sys

from adapters.factory import build_llm
from main import main


def resolve_model_name() -> str:
    """
    Resolve model name from environment or CLI.

    Priority:
    1. CLI argument
    2. LLM_MODEL environment variable
    3. Fail closed
    """
    if len(sys.argv) > 1:
        return sys.argv[1]

    model = os.getenv("LLM_MODEL")
    if model:
        return model

    raise RuntimeError(
        "No model specified. "
        "Pass model as CLI argument or set LLM_MODEL environment variable."
    )


if __name__ == "__main__":
    model_name = resolve_model_name()

    llm = build_llm(model_name)

    main(llm)

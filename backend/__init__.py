"""ASSAY backend package.

Loads `.env` on import. This lives at package import rather than in each entry
point because there are four of them — the FastAPI app, the LangGraph runner,
pytest, and ad-hoc scripts — and a key that works under `uvicorn` but not under a
script is worse than no key loading at all.

`.env.example` has documented `OPENAI_API_KEY` since milestone 2, but nothing ever
read it: `get_client()` goes straight to `os.getenv`, so the only way to supply a
key was to export it in the shell. The documented path silently did nothing, and
the visible symptom was `adjudicator_consulted: 0` on a file with genuinely
low-margin columns — indistinguishable from the margin check being broken.

`load_dotenv` does not override variables already present in the real environment,
so an exported key still wins over the file.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from dotenv import find_dotenv, load_dotenv
except ImportError:  # pragma: no cover - declared in requirements.txt
    logger.debug("python-dotenv is not installed; .env will not be read")
else:
    _dotenv_path = find_dotenv(usecwd=True)
    if _dotenv_path:
        load_dotenv(_dotenv_path)
        logger.debug("loaded environment from %s", _dotenv_path)

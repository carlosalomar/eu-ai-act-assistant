"""Sources, languages and paths shared by the ingestion scripts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
INDEX_DIR = ROOT / "data" / "index"

# EUR-Lex documents are fetched from the Publications Office Cellar, which
# serves the same official XHTML as eur-lex.europa.eu without the bot wall.
CELLAR_URL = "https://publications.europa.eu/resource/celex/{celex}"

# Language code used in this project -> Cellar (ISO 639-2) code.
LANGUAGES = {"en": "eng", "es": "spa", "de": "deu"}

LAWS = {
    "ai_act": {
        "name": {"en": "AI Act", "es": "Reglamento de IA", "de": "KI-Verordnung"},
        "official": "Regulation (EU) 2024/1689",
        # Consolidated text: includes the Digital Omnibus on AI,
        # Regulation (EU) 2026/1744, applicable from 27 July 2026.
        "consolidated_celex": "02024R1689-20260727",
        # Original Official Journal text: the only place the recitals live
        # (consolidated versions drop them).
        "oj_celex": "32024R1689",
    },
    "gdpr": {
        "name": {"en": "GDPR", "es": "RGPD", "de": "DSGVO"},
        "official": "Regulation (EU) 2016/679",
        # Only consolidated version on EUR-Lex; it folds in the 2016 corrigendum.
        "consolidated_celex": "02016R0679-20160504",
        "oj_celex": "32016R0679",
    },
}

# Amending acts kept for reference (not indexed: their changes are already in
# the consolidated text above).
AMENDING_ACTS = {"32026R1744": "Digital Omnibus on AI, Regulation (EU) 2026/1744"}

# Chunks longer than this (in characters) are split at list-item boundaries.
MAX_CHUNK_CHARS = 2500

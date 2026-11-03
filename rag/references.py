"""Find explicit legal references ("Article 6", "artículo 22 del RGPD", "Anhang III").

Used in two places:
  * retrieval: an explicit reference in the question becomes an exact-match
    BM25 token, which dense embeddings cannot provide;
  * evaluation: citations written in free text (the no-RAG baseline) are
    turned into unit ids such as "ai_act:art6" or "gdpr:art22".
"""

import re

LAW_PATTERNS = {
    "gdpr": r"GDPR|RGPD|DSGVO|DS-GVO|2016/679|General Data Protection Regulation|"
            r"Reglamento General de Protección de Datos|Datenschutz-Grundverordnung",
    "ai_act": r"AI\s*Act|AIA|2024/1689|Artificial Intelligence Act|Ley de IA|"
              r"Reglamento de (?:la )?IA|Reglamento de Inteligencia Artificial|"
              r"KI-Verordnung|KI-VO|Verordnung über künstliche Intelligenz",
}
LAW_RE = {law: re.compile(p, re.IGNORECASE) for law, p in LAW_PATTERNS.items()}

NUM = r"\d{1,3}[a-z]?"
CONJ = r"\s*(?:,|and|or|y|o|e|und|oder|bis|to|a)\s*"
ARTICLE_RE = re.compile(
    rf"\b(?:articles?|art[íi]culos?|artikel[n]?|arts?\.?)\s*({NUM}(?:\s*\(\d+\))?"
    rf"(?:{CONJ}{NUM})*)", re.IGNORECASE)
ANNEX_RE = re.compile(r"\b(?:annex(?:es)?|anexos?|anh[äa]ng(?:e)?)\s+([IVXLC]+)\b", re.IGNORECASE)
RECITAL_RE = re.compile(
    r"\b(?:recitals?|considerandos?|erwägungsgr(?:und|ünde|ünden))\s*\(?(\d{1,3})\)?", re.IGNORECASE)

WINDOW = 80  # characters around a reference searched for the law's name


def _law_near(text: str, start: int, end: int, default: str | None) -> str | None:
    """Closest law name to the match, looking mostly after it ("Article 6 GDPR")."""
    best, best_dist = None, None
    for law, regex in LAW_RE.items():
        window_start = max(0, start - WINDOW)
        for m in regex.finditer(text, window_start, min(len(text), end + WINDOW)):
            # Do not jump over a sentence boundary.
            between = text[min(end, m.start()):max(start, m.end())]
            if re.search(r"[.;!?]\s", between):
                continue
            dist = m.start() - end if m.start() >= end else (start - m.end()) * 1.5
            if best_dist is None or dist < best_dist:
                best, best_dist = law, dist
    return best or default


def default_law(text: str) -> str | None:
    """The law mentioned in the text if exactly one is."""
    found = [law for law, regex in LAW_RE.items() if regex.search(text)]
    return found[0] if len(found) == 1 else None


def find_references(text: str, fallback_law: str | None = None) -> list[tuple[str | None, str]]:
    """Return (law or None, unit) pairs, e.g. ("gdpr", "art22"), (None, "anxIII")."""
    law_default = default_law(text) or fallback_law
    refs: list[tuple[str | None, str]] = []
    for m in ARTICLE_RE.finditer(text):
        law = _law_near(text, m.start(), m.end(), law_default)
        for number in re.findall(rf"(?<![(\d]){NUM}(?![\d)])", m.group(1)):
            refs.append((law, f"art{number.lower()}"))
    for m in ANNEX_RE.finditer(text):
        refs.append((_law_near(text, m.start(), m.end(), law_default) or "ai_act",
                     f"anx{m.group(1).upper()}"))
    for m in RECITAL_RE.finditer(text):
        refs.append((_law_near(text, m.start(), m.end(), law_default), f"rct{m.group(1)}"))
    seen, unique = set(), []
    for ref in refs:
        if ref not in seen:
            seen.add(ref)
            unique.append(ref)
    return unique


def reference_units(text: str, fallback_law: str | None = None) -> list[str]:
    """Unit ids ("ai_act:art6") for every reference whose law can be resolved."""
    return [f"{law}:{unit}" for law, unit in find_references(text, fallback_law) if law]

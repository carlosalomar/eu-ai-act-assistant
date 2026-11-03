"""Turn retrieved chunks into the sources shown to the LLM ("small-to-big").

Long paragraphs are indexed point by point (Article 5(1)(a), (b), ...) so that
retrieval is precise, but the model should read the whole paragraph: otherwise
"Which practices are prohibited?" gets an answer built from five of the eight
points. Hits from the same paragraph are therefore merged back into the full
paragraph, in document order, keeping the rank of the best hit.
"""

import re
from dataclasses import replace

from rag.retriever import Hit, get_corpus

MAX_PARAGRAPH_CHARS = 9000  # do not expand beyond this (keeps the prompt within 8k tokens)


def paragraph_key(chunk: dict) -> str | None:
    """'ai_act:art5:p1:a' -> 'ai_act:art5:p1'; None for chunks that are not points."""
    if chunk["kind"] != "article" or not chunk["point"] or chunk["point"].startswith("part"):
        return None
    return chunk["id"].rsplit(":", 1)[0]


def expand(hits: list[Hit], lang: str) -> list[Hit]:
    corpus = get_corpus(lang)
    groups: dict[str, list[dict]] = {}
    for chunk in corpus.chunks:
        key = paragraph_key(chunk)
        if key:
            groups.setdefault(key, []).append(chunk)

    sources, seen = [], set()
    for hit in hits:
        key = paragraph_key(hit.chunk) or hit.chunk["id"]
        if key in seen:
            continue
        seen.add(key)
        members = groups.get(key)
        if not members:
            sources.append(hit)
            continue
        lead = members[0]["text"].split("\n", 1)[0]
        bodies = [m["text"].split("\n", 1)[1] if "\n" in m["text"] else m["text"] for m in members]
        text = "\n".join([lead] + bodies)
        if len(text) > MAX_PARAGRAPH_CHARS:
            sources.append(hit)
            continue
        ref_local = re.sub(r"\([^()]*\)$", "", hit.chunk["ref_local"])
        ref = re.sub(r"\([^()]*\)$", "", hit.chunk["ref"])
        merged = dict(hit.chunk, id=key, point=None, text=text, ref=ref, ref_local=ref_local)
        sources.append(replace(hit, chunk=merged))
    return sources

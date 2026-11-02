"""Split the AI Act and the GDPR into article/paragraph-level chunks.

Every chunk is one legal unit with the metadata needed to cite it:

  * articles   -> one chunk per numbered paragraph ("Article 6(2)"); long
                  paragraphs and long unnumbered articles (e.g. the definitions
                  in Article 3) are split by point ("Article 3(4)", "Article 5(1)(a)")
  * annexes    -> top-level points packed into chunks of at most MAX_CHUNK_CHARS
  * recitals   -> one chunk per recital

Chunk ids are language-independent ("ai_act:art6:p2"), so the same question in
English, Spanish and German has the same gold answer.

Usage: python -m ingest.parse
"""

import json
import re
import warnings
from dataclasses import asdict, dataclass, field

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

from ingest.config import LANGUAGES, LAWS, MAX_CHUNK_CHARS, PROCESSED_DIR, RAW_DIR

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Words used to build the human-readable, language-specific citation.
LABELS = {
    "en": {"article": "Article", "annex": "Annex", "recital": "Recital", "part": "part"},
    "es": {"article": "Artículo", "annex": "Anexo", "recital": "Considerando", "part": "parte"},
    "de": {"article": "Artikel", "annex": "Anhang", "recital": "Erwägungsgrund", "part": "Teil"},
}

SKIP_CLASSES = {"modref", "arrow", "footnote", "title-article-norm", "stitle-article-norm",
                "title-annex-1", "title-annex-2"}


@dataclass
class Chunk:
    id: str                 # language-independent, e.g. "ai_act:art5:p1:a"
    law: str                # "ai_act" | "gdpr"
    lang: str               # "en" | "es" | "de"
    kind: str               # "article" | "annex" | "recital"
    number: str             # "5", "4a", "III", "47"
    paragraph: str | None   # "1" (articles only)
    point: str | None       # "a", "3", "1-4" (annex point range)
    title: str              # article or annex title
    chapter: str | None     # "III"
    chapter_title: str | None
    section: str | None     # "1"
    section_title: str | None
    ref: str                # canonical English citation, e.g. "AI Act, Article 5(1)(a)"
    ref_local: str          # citation in the chunk's language
    celex: str
    text: str               # body only
    text_for_embedding: str = field(default="")  # header + body


# --------------------------------------------------------------------------- #
# Text rendering
# --------------------------------------------------------------------------- #
def clean(text: str) -> str:
    text = text.replace("\xa0", " ").replace("▼B", "").replace("▼M1", "")
    text = re.sub(r"▼[A-Z]\d*", "", text)
    return re.sub(r"\s+", " ", text).strip()


def norm_label(label: str) -> str:
    """'1.' / '(1)' / '(a)' / 'a)' -> '1' / 'a'."""
    return clean(label).strip("().").strip()


def render(el) -> str:
    """Flatten an element to text, keeping list labels ("(a) ...") on new lines."""
    if not isinstance(el, Tag):
        return clean(str(el))
    classes = set(el.get("class", []))
    if classes & SKIP_CLASSES or el.name in {"hr", "br", "col"}:
        return ""
    if el.name == "a" and el.get("href", "").startswith("#E"):  # footnote marker
        return ""
    if "grid-list" in classes:
        label = el.find(class_="grid-list-column-1")
        body = el.find(class_="grid-list-column-2")
        return f"{clean(label.get_text()) if label else ''} {render(body) if body else ''}".strip()
    if el.name == "table" and "grid-list" not in classes:
        rows = [" | ".join(clean(td.get_text(" ")) for td in tr.find_all(["td", "th"]))
                for tr in el.find_all("tr")]
        return "\n".join(r for r in rows if r.strip(" |"))
    if el.name == "span" and "no-parag" in classes:
        return clean(el.get_text())
    parts = [render(child) for child in el.children]
    joiner = "\n" if el.name in {"div", "td", "tbody", "tr"} else " "
    text = joiner.join(p for p in parts if p)
    return re.sub(r"[ \t]+", " ", text).strip()


def is_list_item(el) -> bool:
    return isinstance(el, Tag) and "grid-list" in el.get("class", [])


def list_label(el: Tag) -> str:
    return norm_label(el.find(class_="grid-list-column-1").get_text())


def is_deleted(text: str) -> bool:
    return bool(re.fullmatch(r"[\s—–\-.]*", text))


# --------------------------------------------------------------------------- #
# Articles
# --------------------------------------------------------------------------- #
def article_context(art: Tag) -> dict:
    """Chapter and section that contain an article (from ancestor ids)."""
    ctx = {"chapter": None, "chapter_title": None, "section": None, "section_title": None}
    for anc in art.parents:
        anc_id = anc.get("id", "") if isinstance(anc, Tag) else ""
        heads = anc.find_all("p", class_=["title-division-1", "title-division-2"],
                             recursive=False) if anc_id else []
        title = clean(heads[1].get_text()) if len(heads) > 1 else None
        if re.fullmatch(r"cpt_[IVXLC]+\.sct_\d+", anc_id):
            ctx["section"], ctx["section_title"] = anc_id.split("_")[-1], title
        elif re.fullmatch(r"cpt_[IVXLC]+", anc_id):
            ctx["chapter"], ctx["chapter_title"] = anc_id.split("_")[1], title
    return ctx


def split_by_points(lead: str, items: list[Tag], tail: list[str]) -> list[tuple[str, str]]:
    """One (point, text) pair per list item, each prefixed with the lead-in sentence."""
    out = []
    for item in items:
        body = render(item)
        if is_deleted(re.sub(r"^\S+", "", body)):
            continue
        out.append((list_label(item), f"{lead}\n{body}".strip()))
    if tail and out:
        point, text = out[-1]
        out[-1] = (point, text + "\n" + "\n".join(tail))
    return out


def paragraph_units(art: Tag) -> list[tuple[str | None, str | None, str]]:
    """(paragraph, point, text) units for one article."""
    groups: list[tuple[str | None, list]] = []
    for child in art.children:
        if not isinstance(child, Tag) or child.get("id", "").endswith(".tit_1"):
            continue
        classes = set(child.get("class", []))
        if classes & SKIP_CLASSES:
            continue
        number = child.find("span", class_="no-parag", recursive=False)
        if number:
            groups.append((norm_label(number.get_text()), [child]))
        elif groups:
            groups[-1][1].append(child)
        else:
            groups.append((None, [child]))

    units = []
    for paragraph, elements in groups:
        text = "\n".join(t for t in (render(e) for e in elements) if t)
        text = re.sub(r"^(\(?\d+[a-z]?[.)])\s*\n", r"\1 ", text)  # "2.\nText" -> "2. Text"
        if is_deleted(re.sub(r"^\(?\d+[a-z]?[.)]?", "", text)):
            continue
        if len(text) <= MAX_CHUNK_CHARS:
            units.append((paragraph, None, text))
            continue
        # Too long: split by top-level points, repeating the lead-in sentence.
        container = elements[0].find("div", class_="inline-element", recursive=False) or elements[0]
        scope = list(container.children) if paragraph else elements
        if paragraph and len(elements) > 1:
            scope += elements[1:]
        lead_parts, items, tail = [], [], []
        for el in scope:
            if not isinstance(el, Tag):
                continue
            if is_list_item(el):
                items.append(el)
            elif not items:
                lead_parts.append(render(el))
            else:
                tail.append(render(el))
        lead = (f"{paragraph}. " if paragraph else "") + " ".join(p for p in lead_parts if p)
        pieces = split_by_points(lead.strip(), items, [t for t in tail if t])
        if pieces:
            units.extend((paragraph, point, text) for point, text in pieces)
        else:
            units.extend((paragraph, f"part{i}", t) for i, t in enumerate(hard_split(text), 1))
    return units


def hard_split(text: str) -> list[str]:
    """Last resort: split a long text at line breaks into <= MAX_CHUNK_CHARS pieces."""
    pieces, current = [], ""
    for line in text.split("\n"):
        if current and len(current) + len(line) > MAX_CHUNK_CHARS:
            pieces.append(current)
            current = ""
        current = f"{current}\n{line}".strip()
    return pieces + ([current] if current else [])


def parse_articles(soup, law: str, lang: str, celex: str) -> list[Chunk]:
    chunks = []
    labels = LABELS[lang]
    for art in soup.find_all("div", id=re.compile(r"^art_\d+[a-z]?$")):
        number = art["id"][4:]
        title_el = art.find(class_="stitle-article-norm")
        title = clean(title_el.get_text()) if title_el else ""
        ctx = article_context(art)
        for paragraph, point, text in paragraph_units(art):
            suffix = (f"({paragraph})" if paragraph else "") + (f"({point})" if point else "")
            chunk_id = f"{law}:art{number}" + (f":p{paragraph}" if paragraph else "") \
                + (f":{point}" if point else "")
            chunks.append(Chunk(
                id=chunk_id, law=law, lang=lang, kind="article", number=number,
                paragraph=paragraph, point=point, title=title, **ctx,
                ref=f"{LAWS[law]['name']['en']}, Article {number}{suffix}",
                ref_local=f"{LAWS[law]['name'][lang]}, {labels['article']} {number}{suffix}",
                celex=celex, text=text,
            ))
    return chunks


# --------------------------------------------------------------------------- #
# Annexes
# --------------------------------------------------------------------------- #
def parse_annexes(soup, law: str, lang: str, celex: str) -> list[Chunk]:
    chunks = []
    labels = LABELS[lang]
    for anx in soup.find_all("div", id=re.compile(r"^anx_[IVXLC]+$")):
        number = anx["id"][4:]
        title_el = anx.find(class_="title-annex-2")
        title = clean(title_el.get_text()) if title_el else ""
        # Units: (point label or None, text). Headings stay attached to what follows.
        units, pending = [], ""
        for child in anx.children:
            if not isinstance(child, Tag):
                continue
            text = render(child)
            if not text or is_deleted(text):
                continue
            if "title-gr-seq-level-1" in child.get("class", []) or child.name.startswith("h"):
                pending = f"{pending}\n{text}".strip()
                continue
            label = list_label(child) if is_list_item(child) else None
            units.append((label, f"{pending}\n{text}".strip()))
            pending = ""
        if pending:
            units.append((None, pending))

        # Greedy packing of consecutive units into chunks.
        groups, current, size = [], [], 0
        for unit in units:
            if current and size + len(unit[1]) > MAX_CHUNK_CHARS:
                groups.append(current)
                current, size = [], 0
            current.append(unit)
            size += len(unit[1])
        if current:
            groups.append(current)

        seen: set = set()
        for i, group in enumerate(groups, 1):
            points = [label for label, _ in group if label]
            point = None
            if len(groups) > 1:
                point = (f"{points[0]}-{points[-1]}" if len(points) > 1 else points[0]) \
                    if points else f"part{i}"
            text = "\n".join(t for _, t in group)
            for piece_no, piece in enumerate(hard_split(text) if len(text) > MAX_CHUNK_CHARS * 1.5
                                             else [text], 1):
                piece_point = point if piece_no == 1 else f"{point or ''}~{piece_no}"
                if piece_point in seen:  # repeated labels, e.g. Annex X restarts at (a)
                    piece_point = f"{piece_point}~g{i}"
                seen.add(piece_point)
                suffix = f", point {piece_point}" if piece_point else ""
                chunks.append(Chunk(
                    id=f"{law}:anx{number}" + (f":{piece_point}" if piece_point else ""),
                    law=law, lang=lang, kind="annex", number=number, paragraph=None,
                    point=piece_point, title=title, chapter=None, chapter_title=None,
                    section=None, section_title=None,
                    ref=f"{LAWS[law]['name']['en']}, Annex {number}{suffix}",
                    ref_local=f"{LAWS[law]['name'][lang]}, {labels['annex']} {number}{suffix}",
                    celex=celex, text=piece,
                ))
    return chunks


# --------------------------------------------------------------------------- #
# Recitals (original Official Journal text)
# --------------------------------------------------------------------------- #
def parse_recitals(soup, law: str, lang: str, celex: str) -> list[Chunk]:
    chunks = []
    labels = LABELS[lang]
    for rct in soup.find_all("div", id=re.compile(r"^rct_\d+$")):
        number = rct["id"][4:]
        cells = rct.find_all("td")
        body = cells[-1] if cells else rct
        text = clean(body.get_text(" "))
        chunks.append(Chunk(
            id=f"{law}:rct{number}", law=law, lang=lang, kind="recital", number=number,
            paragraph=None, point=None, title="", chapter=None, chapter_title=None,
            section=None, section_title=None,
            ref=f"{LAWS[law]['name']['en']}, Recital {number}",
            ref_local=f"{LAWS[law]['name'][lang]}, {labels['recital']} {number}",
            celex=celex, text=text,
        ))
    return chunks


# --------------------------------------------------------------------------- #
def embedding_header(chunk: Chunk) -> str:
    """Short context line prepended before embedding / BM25 indexing."""
    parts = [chunk.ref_local]
    if chunk.title:
        parts.append(chunk.title)
    if chunk.chapter_title:
        parts.append(chunk.chapter_title)
    return " — ".join(parts)


def load(path):
    return BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for lang in LANGUAGES:
        chunks: list[Chunk] = []
        for law, cfg in LAWS.items():
            consolidated = load(RAW_DIR / f"{cfg['consolidated_celex']}_{lang}.html")
            original = load(RAW_DIR / f"{cfg['oj_celex']}_{lang}.html")
            chunks += parse_articles(consolidated, law, lang, cfg["consolidated_celex"])
            chunks += parse_annexes(consolidated, law, lang, cfg["consolidated_celex"])
            chunks += parse_recitals(original, law, lang, cfg["oj_celex"])
        for chunk in chunks:
            chunk.text_for_embedding = f"{embedding_header(chunk)}\n{chunk.text}"
        ids = [c.id for c in chunks]
        dupes = {i for i in ids if ids.count(i) > 1}
        assert not dupes, f"duplicate chunk ids in {lang}: {sorted(dupes)[:10]}"
        out = PROCESSED_DIR / f"chunks_{lang}.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        print(f"{lang}: {len(chunks)} chunks -> {out.relative_to(PROCESSED_DIR.parents[1])}")


if __name__ == "__main__":
    main()

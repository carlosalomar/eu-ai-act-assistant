"""Download the official AI Act and GDPR texts from EUR-Lex (Cellar).

For each law and language we keep two documents:
  * the latest consolidated version (articles + annexes in force), and
  * the original Official Journal publication (recitals).

Before downloading, the script asks the EUR-Lex SPARQL endpoint for the newest
consolidated version and for any amending act, so a new consolidation is not
missed silently.

Usage: python -m ingest.download
"""

import csv
import io
import json
import sys

import requests

from ingest.config import AMENDING_ACTS, CELLAR_URL, LANGUAGES, LAWS, RAW_DIR

SPARQL_URL = "https://publications.europa.eu/webapi/rdf/sparql"
TIMEOUT = 120

CONSOLIDATED_QUERY = """
PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
SELECT DISTINCT ?celex WHERE {
  ?w cdm:resource_legal_id_celex ?celex .
  FILTER(STRSTARTS(STR(?celex), "0%s"))
} ORDER BY ?celex
"""

AMENDMENTS_QUERY = """
PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
SELECT DISTINCT ?celex WHERE {
  ?t cdm:resource_legal_id_celex "%s"^^<http://www.w3.org/2001/XMLSchema#string> .
  ?w cdm:resource_legal_amends_resource_legal ?t ;
     cdm:resource_legal_id_celex ?celex .
}
"""


def sparql(query: str) -> list[str]:
    response = requests.post(
        SPARQL_URL, data={"query": query}, headers={"Accept": "text/csv"}, timeout=TIMEOUT
    )
    response.raise_for_status()
    rows = list(csv.reader(io.StringIO(response.text)))
    return [row[0] for row in rows[1:]]


def check_versions() -> dict:
    """Warn if EUR-Lex has a newer consolidation or amendment than the config."""
    report = {}
    for key, law in LAWS.items():
        base = law["oj_celex"][1:]  # "2024R1689"
        consolidated = sparql(CONSOLIDATED_QUERY % base)
        amendments = sparql(AMENDMENTS_QUERY % law["oj_celex"])
        latest = max(consolidated) if consolidated else None
        report[key] = {"consolidated_versions": consolidated, "amending_acts": amendments}
        if latest and latest != law["consolidated_celex"]:
            print(f"WARNING {key}: newest consolidation is {latest}, "
                  f"config uses {law['consolidated_celex']}", file=sys.stderr)
        unknown = set(amendments) - set(AMENDING_ACTS)
        if unknown:
            print(f"WARNING {key}: unreviewed amending acts {sorted(unknown)}", file=sys.stderr)
    return report


def download(celex: str, lang: str) -> None:
    target = RAW_DIR / f"{celex}_{lang}.html"
    if target.exists():
        return
    response = requests.get(
        CELLAR_URL.format(celex=celex),
        headers={"Accept": "application/xhtml+xml, text/html;q=0.9",
                 "Accept-Language": LANGUAGES[lang]},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    target.write_bytes(response.content)
    print(f"downloaded {target.name} ({len(response.content) / 1e6:.1f} MB)")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    report = check_versions()
    (RAW_DIR / "versions.json").write_text(json.dumps(report, indent=2))
    for law in LAWS.values():
        for lang in LANGUAGES:
            download(law["consolidated_celex"], lang)
            download(law["oj_celex"], lang)


if __name__ == "__main__":
    main()

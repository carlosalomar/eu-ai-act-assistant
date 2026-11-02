# Data sources

All legal texts come from the official EUR-Lex publications of the Publications
Office of the European Union. They are fetched from the Cellar repository
(`https://publications.europa.eu/resource/celex/<CELEX>`), which serves the same
XHTML documents as eur-lex.europa.eu and can be scripted (eur-lex.europa.eu
sits behind a bot challenge).

| Law | Text used for articles and annexes | Text used for recitals |
|---|---|---|
| AI Act, Regulation (EU) 2024/1689 | Consolidated version `02024R1689-20260727` | Original OJ publication `32024R1689` |
| GDPR, Regulation (EU) 2016/679 | Consolidated version `02016R0679-20160504` | Original OJ publication `32016R0679` |

Languages: English, Spanish and German.

## Versions and amendments (checked 2 November 2026)

`python -m ingest.download` queries the EUR-Lex SPARQL endpoint before
downloading and writes the result to `data/raw/versions.json`. It prints a
warning if a newer consolidated version or an amending act appears.

**AI Act.** Two consolidated versions exist: the original text (12 July 2024)
and the version of 27 July 2026, which incorporates the **Digital Omnibus on AI**,
Regulation (EU) 2026/1744 of 8 July 2026. That amendment, among other changes,
inserts Articles 4a, 60a and 75a-75d and Annex XIV, adds new prohibited
practices to Article 5(1) and changes the timeline in Article 113. The project
indexes the 2026 consolidated text, so answers reflect the law as amended.
Four corrigenda to the original text were published between October 2025 and
May 2026 (`32024R1689R(01)`-`R(04)`).

**GDPR.** No amending act. The only consolidated version (4 May 2016) folds in
the first corrigendum; later corrigenda (2018, 2021) only fix individual
language versions.

## Why recitals come from a different document

Consolidated versions contain only the enacting terms (articles) and annexes;
recitals are dropped. Recitals still matter for interpretation, so they are
taken from the original Official Journal text. The Digital Omnibus did not
amend existing recitals of the AI Act (an amending act has its own recitals,
which are not indexed).

## Disclaimer

Consolidated texts are documentation tools without legal effect; only the
Official Journal versions are authentic. This assistant is a portfolio
project, not legal advice.

"""Gradio demo: ask the AI Act / GDPR assistant a question and see its sources.

The LLM is any OpenAI-compatible server (vLLM on Colab, or llama.cpp locally),
configured with LLM_BASE_URL and LLM_MODEL, see README.

Usage: python app.py            # http://localhost:7860
       python app.py --share    # public link (e.g. from Colab)
"""

import argparse
import re
import time

import gradio as gr

from rag.lang import detect_language
from rag.pipeline import Assistant

MODES = {
    "RAG: hybrid search + reranker": "hybrid_rerank",
    "RAG: vector search": "dense",
    "LLM only (no retrieval)": "llm_only",
}
EURLEX_LANG = {"en": "EN", "es": "ES", "de": "DE"}
EXAMPLES = [
    "Which AI practices are prohibited?",
    "¿Un sistema de selección de personal con IA es de alto riesgo?",
    "Welche Pflichten haben Anbieter von KI-Modellen mit allgemeinem Verwendungszweck?",
    "Within how many hours must a personal data breach be notified?",
    "What is the corporate tax rate in Ireland?",
]

_assistants: dict[str, Assistant] = {}


def assistant(mode: str) -> Assistant:
    if mode not in _assistants:
        _assistants[mode] = Assistant(mode)
    return _assistants[mode]


def eurlex_url(chunk: dict, lang: str) -> str:
    """Link to the article, annex or recital on EUR-Lex."""
    anchor = {"article": "art_", "annex": "anx_", "recital": "rct_"}[chunk["kind"]] + chunk["number"]
    return (f"https://eur-lex.europa.eu/legal-content/{EURLEX_LANG[lang]}/TXT/"
            f"?uri=CELEX:{chunk['celex']}#{anchor}")


def link_citations(text: str, sources, lang: str) -> str:
    """Turn [n] markers into links to the cited provision."""
    def replace(match):
        index = int(match.group(1)) - 1
        if not 0 <= index < len(sources):
            return match.group(0)
        chunk = sources[index].chunk
        return f"[[{index + 1}]]({eurlex_url(chunk, lang)} \"{chunk['ref_local']}\")"
    return re.sub(r"\[(\d+)\]", replace, text)


def format_sources(sources, cited_units: list[str], lang: str) -> str:
    if not sources:
        return "_No retrieval in this mode: the model answers from memory._"
    blocks = []
    for i, hit in enumerate(sources, 1):
        chunk = hit.chunk
        mark = "**cited** · " if hit.unit in cited_units else ""
        score = f"reranker {hit.rerank_score:.2f}" if hit.rerank_score is not None else ""
        title = f" — {chunk['title']}" if chunk["title"] else ""
        text = chunk["text"] if len(chunk["text"]) < 700 else chunk["text"][:700] + "…"
        blocks.append(f"**[{i}] [{chunk['ref_local']}]({eurlex_url(chunk, lang)}){title}**  \n"
                      f"<sub>{mark}{score}</sub>\n\n{text}")
    return "\n\n---\n\n".join(blocks)


def answer(question: str, mode_label: str):
    question = (question or "").strip()
    if not question:
        return "Type a question first.", "", ""
    mode = MODES[mode_label]
    lang = detect_language(question)
    start = time.time()
    try:
        result = assistant(mode).ask(question, lang)
    except Exception as error:  # LLM server down: still show what retrieval found
        sources = assistant(mode).retrieve(question, lang) if mode != "llm_only" else []
        message = f"⚠️ Could not reach the LLM ({type(error).__name__}). Is the server running?"
        return message, format_sources(sources, [], lang), ""
    body = result.display_text
    if mode != "llm_only" and not result.abstained:
        body = link_citations(body, result.sources, lang)
    if result.cited_refs:
        body += "\n\n**Cited:** " + " · ".join(result.cited_refs)
    status = (f"Language: {lang.upper()} · {mode_label} · {time.time() - start:.1f}s"
              + (" · abstained" if result.abstained else ""))
    return body, format_sources(result.sources, result.cited_units, lang), status


def build() -> gr.Blocks:
    with gr.Blocks(title="EU AI Act & GDPR assistant") as demo:
        gr.Markdown(
            "# EU AI Act & GDPR assistant\n"
            "Answers questions about Regulation (EU) 2024/1689 (as amended by the Digital "
            "Omnibus on AI) and the GDPR, in English, Spanish or German, citing the exact "
            "provision. If the two regulations do not answer the question, it says so. "
            "*Portfolio project, not legal advice.*")
        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(label="Question", lines=2,
                                      placeholder="e.g. Is emotion recognition at work allowed?")
            with gr.Column(scale=2):
                mode = gr.Radio(list(MODES), value=list(MODES)[0], label="System")
        ask = gr.Button("Ask", variant="primary")
        gr.Examples(EXAMPLES, inputs=question)
        status = gr.Markdown()
        output = gr.Markdown(label="Answer")
        with gr.Accordion("Sources given to the model", open=False):
            sources = gr.Markdown()
        for trigger in (ask.click, question.submit):
            trigger(answer, inputs=[question, mode], outputs=[output, sources, status])
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build().launch(share=args.share, server_port=args.port)

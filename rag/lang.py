"""Tiny stop-word based language detector for en / es / de questions.

Questions are short and only three languages are supported, so counting
frequent function words is enough and avoids an extra dependency.
"""

import re

MARKERS = {
    "en": set("the is are what which who how does do of and to for an a in can must should "
              "my our it this that with".split()),
    "es": set("el la los las qué cuál cuáles cómo es son del de y para un una en que se "
              "puede debe mi nuestra con por lo".split()),
    "de": set("der die das ist sind was welche wie und für ein eine nach muss kann im "
              "mit von zu des den dem wer darf".split()),
}
SPECIAL_CHARS = {"es": "ñ¿¡áéíóú", "de": "äöüß"}


def detect_language(text: str, default: str = "en") -> str:
    words = re.findall(r"\w+", text.lower())
    scores = {lang: sum(w in markers for w in words) for lang, markers in MARKERS.items()}
    for lang, chars in SPECIAL_CHARS.items():
        scores[lang] += 2 * sum(text.lower().count(c) for c in chars)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else default

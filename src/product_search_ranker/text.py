"""Tokenizer, HTML stripping and query-slice regexes for English and Spanish."""

import re
import unicodedata

TAG_RE = re.compile(r"<[^>]*>")
SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[^\W_]+")

# bm25s ships the English list; Spanish is short on purpose, no stemming either way.
STOPWORDS_ES = frozenset(
    "a al con de del el ella en es la las lo los para por que se sin su un una y o".split()
)

NEGATION_EN = {"without", "no", "not", "non"}
NEGATION_ES = {"sin", "no", "excepto"}
UNITS = (
    "oz|ml|l|lb|lbs|kg|g|mg|gb|tb|mb|inch|in|cm|mm|m|ft|pack|pk|count|ct|pcs|piezas|pulgadas|"
    "litros|gramos|unidades|w|v|mah|hz"
)
NUMBER_OR_UNIT_RE = re.compile(rf"\d|\b({UNITS})\b", re.IGNORECASE)

LENGTH_SLICES = ("len_1_2", "len_3_5", "len_6_plus")
SLICES = (*LENGTH_SLICES, "negation", "number_or_unit", "brand_in_query")
HARD_SLICES = ("negation", "number_or_unit", "brand_in_query")


def strip_html(s: str | None) -> str:
    if not s:
        return ""
    s = TAG_RE.sub(" ", s).replace("&nbsp;", " ").replace("&amp;", "&")
    return SPACE_RE.sub(" ", s).strip()


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    # Strip accents so "cordón" and "cordon" match; the ñ -> n loss is accepted.
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def tokenize(s: str | None, locale: str = "us") -> list[str]:
    tokens = TOKEN_RE.findall(normalize(s or ""))
    stop = STOPWORDS_ES if locale == "es" else _english_stopwords()
    return [t for t in tokens if t not in stop]


def _english_stopwords() -> frozenset[str]:
    global _EN
    if _EN is None:
        import bm25s.stopwords

        _EN = frozenset(bm25s.stopwords.STOPWORDS_EN)
    return _EN


_EN: frozenset[str] | None = None


def query_slices(query: str, locale: str, brands: list[str] | None = None) -> set[str]:
    """Slices a query belongs to. `brands` are the brand strings of its candidates."""
    tokens = TOKEN_RE.findall(normalize(query))
    out = set()
    n = len(tokens)
    out.add("len_1_2" if n <= 2 else "len_3_5" if n <= 5 else "len_6_plus")
    neg = NEGATION_ES if locale == "es" else NEGATION_EN
    if neg.intersection(tokens):
        out.add("negation")
    if NUMBER_OR_UNIT_RE.search(normalize(query)):
        out.add("number_or_unit")
    q = " " + " ".join(tokens) + " "
    for b in brands or []:
        bt = " ".join(TOKEN_RE.findall(normalize(b or "")))
        if bt and f" {bt} " in q:
            out.add("brand_in_query")
            break
    return out

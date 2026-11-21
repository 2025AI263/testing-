"""Toxicity classifier with two swappable backends.

1. lexicon     - offline, no ML dependencies. WHOLE-WORD keyword matching plus a few
                 normalisations (stretched letters, leetspeak, f*ck) and simple context
                 rules (negation, insults aimed at "you"). It is a keyword scorer, NOT a
                 trained model: it misses sarcasm and implied abuse and will have some
                 false positives. Validate it on a hand-labelled sample before trusting rates.
2. transformer - unitary/toxic-bert via detoxify (pip install detoxify torch).

Both expose: predict(text) -> {"toxicity": float in [0,1], "is_toxic": bool,
                               "categories": {cat: float}}
and predict_many(list_of_texts) -> list of the same dicts.
"""
import re, html
import config

# ----------------------------- lexicon backend ------------------------------
# category: (weight per distinct term hit, accept suffixes, tolerate f*ck-style obfuscation, terms)
LEXICON = {
    "profanity": (0.35, True, True, ["fuck", "shit", "bitch", "asshole", "dumbass",
                                     "dickhead", "dick", "cunt", "piss", "bastard"]),
    "insult":    (0.30, True, True, ["idiot", "stupid", "dumb", "moron", "loser",
                                     "imbecile", "scumbag", "shut up", "hate you"]),
    # Deliberately strict: "hurt you" / "beat you" show up in song lyrics and game banter.
    "threat":    (0.80, False, False, ["kill yourself", "kys", "go die", "hope you die",
                                       "kill you", "beat you up", "i will hurt you",
                                       "ill hurt you", "come find you", "find you and"]),
    "hate":      (0.90, True, True, ["nigger", "faggot", "retard"]),
    "spam":      (0.25, False, False, ["subscribe to my channel", "check my channel",
                                       "check out my channel", "free money", "click the link",
                                       "crypto giveaway", "buy now", "promo code"]),
}
SPAM_LINK_WEIGHT = 0.25

# Words that are only abusive when the comment addresses someone ("you", "your", "ur" ...).
# "take out the trash" or "ugly sweater" must not count.
DIRECTED_INSULTS = ["trash", "clown", "useless", "worthless", "ugly", "pathetic", "nazi"]
DIRECTED_WEIGHT = 0.30

_YOU = {"you", "your", "youre", "you're", "you've", "you'll", "yourself", "ur", "u", "yall", "y'all"}
_NEG = {"not", "never", "nobody", "dont", "don't", "isnt", "isn't", "arent", "aren't",
        "wasnt", "wasn't", "nothing", "nor", "cant", "can't"}

_WORD = re.compile(r"[a-z0-9']+")
_TOKEN = re.compile(r"[a-z0-9]*[a-z][a-z0-9]*")         # tokens that contain a letter
_LEET = str.maketrans("013457", "oieast")
_URL = re.compile(r"https?://|www\.")
_SUFFIX = r"(?:s|es|ed|er|ers|ing|in|y|ty)?"
_VOWELS = set("aeiou")


def _collapse(s: str) -> str:
    """Squash repeated characters ('fuckkkk' -> 'fuck'). Applied to text AND terms."""
    return re.sub(r"(.)\1+", r"\1", s)


def _normalise(raw: str) -> str:
    t = raw.lower()
    t = _TOKEN.sub(lambda m: m.group(0).translate(_LEET), t)   # idi0t -> idiot (digits only)
    return _collapse(t)


def _pattern(term: str, suffix: bool, obfuscate: bool) -> "re.Pattern":
    parts = []
    for ch in _collapse(term):
        if ch == " ":
            parts.append(r"\s+")
        elif obfuscate and ch in _VOWELS:
            parts.append(f"(?:{ch}|[*#@!])")                    # f*ck, sh!t
        else:
            parts.append(re.escape(ch))
    return re.compile(r"(?<![a-z0-9])" + "".join(parts) + (_SUFFIX if suffix else "") + r"(?![a-z0-9])")


def _build_entries():
    entries = []
    for cat, (weight, suffix, obfuscate, terms) in LEXICON.items():
        for term in terms:
            entries.append(dict(cat=cat, term=term, weight=weight, directed_only=False,
                                rx=_pattern(term, suffix, obfuscate)))
    for term in DIRECTED_INSULTS:
        entries.append(dict(cat="insult", term=term, weight=DIRECTED_WEIGHT, directed_only=True,
                            rx=_pattern(term, True, False)))
    return entries


_ENTRIES = _build_entries()


def _negated(t: str, pos: int) -> bool:
    """True if a negation word appears in the 5 words before position `pos`."""
    prev = _WORD.findall(t[max(0, pos - 50):pos])[-5:]
    return any(w in _NEG for w in prev)


_AFTER = re.compile(r"\s+([a-z0-9']+)")


def _addressed(t: str, m) -> bool:
    """True if 'you'/'your'/'ur' is within 3 words BEFORE the match (same sentence) or is the
    very next word with only whitespace in between.

    Proximity matters: "you stupid idiot" is aimed at someone, "this laptop is stupid, if you
    use a mouse ..." or "it's useless. You can ..." are not.
    """
    before = re.split(r"[.!?\n]", t[max(0, m.start() - 40):m.start()])[-1]
    if any(w in _YOU for w in _WORD.findall(before)[-3:]):
        return True
    nxt = _AFTER.match(t, m.end())
    return bool(nxt and nxt.group(1) in _YOU)


def _lexicon_score(text: str) -> dict:
    raw = html.unescape(text or "")
    t = _normalise(raw)

    hits = {}                                   # (category, term) -> weight; each term counts once
    directed = False                            # is any abusive hit aimed at "you"?
    for e in _ENTRIES:
        for m in e["rx"].finditer(t):
            if e["cat"] == "insult" and _negated(t, m.start()):
                continue                        # "this is not stupid"
            aimed = e["term"].endswith(" you") or _addressed(t, m)
            if e["directed_only"] and not aimed:
                continue                        # "ugly sweater", "take out the trash"
            hits[(e["cat"], e["term"])] = e["weight"]
            directed = directed or (aimed and e["cat"] in ("profanity", "insult", "hate"))
            break
    if _URL.search(raw.lower()):
        hits[("spam", "<link>")] = SPAM_LINK_WEIGHT

    keep = {c: 1.0 for c in LEXICON}            # P(no hit) per category (noisy-OR)
    for (cat, _), w in hits.items():
        keep[cat] *= (1.0 - w)
    cats = {c: round(1.0 - v, 4) for c, v in keep.items()}

    all_keep = 1.0
    for v in keep.values():
        all_keep *= v
    score = 1.0 - all_keep

    if hits:                                    # style signals only matter when something was found
        if directed:
            score += 0.20                       # abuse aimed at someone
        letters = [c for c in raw if c.isalpha()]
        if len(letters) > 10 and sum(c.isupper() for c in letters) / len(letters) > 0.6:
            score += 0.10
        if raw.count("!") > 4:
            score += 0.10
    return {"toxicity": round(min(score, 1.0), 4), "categories": cats}


class LexiconBackend:
    def predict(self, text: str) -> dict:
        return _lexicon_score(text)


# --------------------------- transformer backend ----------------------------
class TransformerBackend:
    def __init__(self):
        from detoxify import Detoxify           # lazy import: only if selected
        self._model = Detoxify("original")

    @staticmethod
    def _row(scores: dict, i=None) -> dict:
        get = (lambda v: float(v)) if i is None else (lambda v: float(v[i]))
        cats = {k.replace("_", " "): round(get(v), 4) for k, v in scores.items() if k != "toxicity"}
        return {"toxicity": round(get(scores["toxicity"]), 4), "categories": cats}

    def predict(self, text: str) -> dict:
        return self._row(self._model.predict(text[:2000]))

    def predict_batch(self, texts, chunk=64) -> list:
        rows = []
        for i in range(0, len(texts), chunk):
            part = [t[:2000] for t in texts[i:i + chunk]]
            scores = self._model.predict(part)
            rows += [self._row(scores, j) for j in range(len(part))]
        return rows


# ------------------------------- dispatcher ---------------------------------
_backend = None


def get_backend():
    global _backend
    if _backend is None:
        _backend = TransformerBackend() if config.TOXICITY_BACKEND == "transformer" else LexiconBackend()
    return _backend


def _finish(r: dict) -> dict:
    r["is_toxic"] = bool(r["toxicity"] >= config.TOXIC_THRESHOLD)
    return r


def predict(text: str) -> dict:
    return _finish(get_backend().predict(text or ""))


def predict_many(texts) -> list:
    backend = get_backend()
    texts = [t or "" for t in texts]
    rows = backend.predict_batch(texts) if hasattr(backend, "predict_batch") \
        else [backend.predict(t) for t in texts]
    return [_finish(r) for r in rows]

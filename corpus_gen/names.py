"""Deterministic fictional names and canary codes.

Invariants (checked by `check_names`):
  * every entity has exactly one distinctive token, unique across the whole pool
    (case-insensitive), so no canonical name is a substring of another and
    LightRAG's exact, case-sensitive name merge never conflates two entities;
  * distinctive tokens are invented (syllable-built), >= 6 letters, not English
    words (english-words 'web2'), absent from real web text (wordfreq zipf == 0,
    which rejects real surnames, places, characters, brands), and not blocklisted;
  * canary codes match `CODE_REGEX` and nothing else in the generator emits that shape.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .config import rng_for

CODE_REGEX = re.compile(r"\b[A-Z]{2}-\d{4}-[A-Z]\b")

# Pronounceable CV(C) syllables: odd spellings invite the extraction LLM to
# "correct" names, which splits one entity into two LightRAG nodes.
_ONSETS = ["b", "d", "f", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z",
           "br", "dr", "tr", "st", "th", "kr"]
_NUCLEI = ["a", "e", "i", "o", "u", "a", "e", "o"]
_CODAS = ["", "", "", "", "n", "r", "l", "s", "m"]
_ENDINGS = ["", "", "a", "en", "is", "or", "an", "el", "ine", "ett"]

# Real words / brands that syllable generation could plausibly hit.
_BLOCKLIST = {
    "google", "amazon", "oracle", "tesla", "meta", "apple", "intel", "nvidia", "cisco",
    "adobe", "zoom", "slack", "stripe", "shell", "exxon", "boeing", "pfizer", "honda",
    "toyota", "nokia", "sony", "samsung", "ford", "visa", "kraft", "nestle", "delta",
    "verizon", "prada", "gucci", "rolex", "lexus", "kodak", "xerox", "dell", "lenovo",
    "panda", "salt", "bread", "brand", "trend", "stand", "grand", "brain", "drain",
    "train", "stern", "storm", "steal", "steel", "thread", "thorn", "shore", "share",
    "saturn", "venus", "mars", "terra", "nova", "luna", "sola", "zeus", "thor",
    "halo", "titan", "atlas", "orion", "phoenix", "falcon", "raven", "kraken",
    # famous fictional / brand names not in dictionaries
    "vader", "sauron", "gandalf", "frodo", "zelda", "mario", "sonic", "pikachu",
    "hermione", "narnia", "gotham", "wakanda", "tatooine", "hogwarts", "skynet",
    "cortana", "siri", "alexa", "tinder", "uber", "lyft", "hulu", "roku", "zara",
    "kia", "nissan", "subaru", "mazda", "audi", "volvo", "skoda", "dacia", "lada",
    "fanta", "sprite", "pepsi", "lego", "ikea", "nike", "puma", "fila", "asus", "acer",
}


def _load_dictionary() -> set[str]:
    """English wordlist (english-words 'web2'); empty set if not installed."""
    try:
        from english_words import get_english_words_set
        return get_english_words_set(["web2"], lower=True)
    except Exception:  # noqa: BLE001
        return set()


_DICTIONARY = _load_dictionary()


def word_frequency(w: str) -> float:
    """Zipf frequency of `w` in English web text (wordfreq); 0.0 if unknown or not installed.
    Any nonzero value means the token occurs in real text (names, places, brands)."""
    try:
        from wordfreq import zipf_frequency
        return zipf_frequency(w.lower(), "en")
    except Exception:  # noqa: BLE001
        return 0.0
# Codes: avoid 2-letter prefixes that read as common acronyms.
_CODE_PREFIX_BLOCK = {"HR", "IT", "QA", "PR", "US", "UK", "EU", "AI", "ML", "ID", "OK",
                      "TV", "PC", "CV", "PM", "AM", "VP", "CF", "OS", "IP", "DB", "UI"}

PERSON_TITLES = ["Director of Engineering", "Program Manager", "Staff Engineer",
                 "Finance Lead", "Head of Security", "Product Manager", "Data Scientist",
                 "Operations Manager", "Legal Counsel", "Research Lead",
                 "Procurement Manager", "Chief of Staff", "Solutions Architect",
                 "Compliance Officer", "Site Reliability Lead", "UX Researcher"]
PRODUCT_NOUNS = ["Ledger", "Beacon", "Console", "Relay", "Vault", "Compass", "Lattice",
                 "Harbor", "Prism", "Loom"]
TEAM_SUFFIXES = ["Platform Group", "Analytics Unit", "Infrastructure Team",
                 "Applied Research Lab", "Risk Office", "Integration Team"]
VENDOR_SUFFIXES = ["Systems", "Partners", "Logistics", "Labs", "Fabrication", "Consulting"]
SITE_SUFFIXES = ["Campus", "Data Center", "Field Office", "Annex"]
COMPANY_SUFFIXES = ["Dynamics", "Industries", "Technologies", "Holdings"]
CANARY_KINDS = ["budget code", "cost center", "contract number", "access badge series",
                "procurement order", "escrow reference", "audit finding ID", "vault key ID"]

ENTITY_TYPES = ("person", "project", "product", "team", "vendor", "site")


@dataclass(frozen=True)
class EntityName:
    canonical: str     # exact surface form that must appear verbatim in docs
    distinctive: str   # unique token, for collision checks
    type: str
    title: str | None = None  # person job title


class NameFactory:
    def __init__(self, seed: int):
        self.rng = rng_for(seed, "names")
        self.used: set[str] = set()

    def _syllable(self) -> str:
        r = self.rng
        return r.choice(_ONSETS) + r.choice(_NUCLEI) + r.choice(_CODAS)

    def word(self, min_len: int = 6, max_len: int = 9) -> str:
        """Fresh invented token, unique (case-insensitive) within this factory."""
        for _ in range(10_000):
            w = "".join(self._syllable() for _ in range(self.rng.choice([2, 2, 3])))
            w = (w + self.rng.choice(_ENDINGS)).capitalize()
            lw = w.lower()
            if re.search(r"[aeiou]{3}|[^aeiou]{3}|([aeiou])\1", lw):  # pileups, doubled vowels
                continue
            if not (min_len <= len(w) <= max_len):
                continue
            if lw in self.used or lw in _BLOCKLIST or lw in _DICTIONARY:
                continue
            if word_frequency(lw) > 0.0:  # occurs in real text: a real name/place/brand
                continue
            # no token may contain / be contained in another (substring safety)
            if any(lw in u or u in lw for u in self.used):
                continue
            if re.search(r"(.)\1\1", lw):  # no triple letters
                continue
            self.used.add(lw)
            return w
        raise RuntimeError("name space exhausted; widen syllable lists")

    # Entities --------------------------------------------------------------
    def person(self) -> EntityName:
        first, last = self.word(4, 7), self.word(6, 9)
        # distinctive = surname; first name is also reserved via self.used
        return EntityName(f"{first} {last}", last, "person", self.rng.choice(PERSON_TITLES))

    def project(self) -> EntityName:
        w = self.word()
        return EntityName(f"Project {w}", w, "project")

    def product(self) -> EntityName:
        w = self.word()
        return EntityName(f"{w} {self.rng.choice(PRODUCT_NOUNS)}", w, "product")

    def team(self) -> EntityName:
        w = self.word()
        return EntityName(f"{w} {self.rng.choice(TEAM_SUFFIXES)}", w, "team")

    def vendor(self) -> EntityName:
        w = self.word()
        return EntityName(f"{w} {self.rng.choice(VENDOR_SUFFIXES)}", w, "vendor")

    def site(self) -> EntityName:
        w = self.word()
        return EntityName(f"{w} {self.rng.choice(SITE_SUFFIXES)}", w, "site")

    def company(self) -> str:
        return f"{self.word(6, 9)} {self.rng.choice(COMPANY_SUFFIXES)}"


def make_names(seed: int, counts: dict[str, int]) -> tuple[str, dict[str, list[EntityName]]]:
    """Company name + entity names per type. Deterministic in (seed, counts order)."""
    f = NameFactory(seed)
    company = f.company()
    out: dict[str, list[EntityName]] = {}
    for etype in ENTITY_TYPES:
        gen = getattr(f, etype)
        out[etype] = [gen() for _ in range(counts.get(etype, 0))]
    check_names(company, [n for ns in out.values() for n in ns])
    return company, out


def check_names(company: str, names: list[EntityName]) -> None:
    canon = [n.canonical for n in names] + [company]
    low = [c.lower() for c in canon]
    if len(set(low)) != len(low):
        raise AssertionError("duplicate canonical names")
    for i, a in enumerate(low):
        for j, b in enumerate(low):
            if i != j and a in b:
                raise AssertionError(f"name {canon[i]!r} is a substring of {canon[j]!r}")
    dist = [n.distinctive.lower() for n in names]
    if len(set(dist)) != len(dist):
        raise AssertionError("distinctive tokens not unique")
    for n in names:
        if CODE_REGEX.search(n.canonical):
            raise AssertionError(f"entity name looks like a canary code: {n.canonical}")


# Canary codes --------------------------------------------------------------
class CanaryCodeFactory:
    def __init__(self, seed: int):
        self.rng = rng_for(seed, "canary_codes")
        self.used: set[str] = set()
        self.used_prefixes: set[str] = set()

    def code(self) -> str:
        r = self.rng
        letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # no I/O (avoid 1/0 confusion)
        for _ in range(10_000):
            prefix = r.choice(letters) + r.choice(letters)
            if prefix in _CODE_PREFIX_BLOCK or prefix in self.used_prefixes:
                continue
            digits = f"{r.randrange(1000, 10000)}"
            if len(set(digits)) < 3:  # avoid 1111 / 2020-ish strings
                continue
            if digits.startswith(("19", "20")):  # avoid year-like numbers
                continue
            c = f"{prefix}-{digits}-{r.choice(letters)}"
            if c in self.used:
                continue
            # unique prefix per canary => partial matches stay attributable
            self.used.add(c)
            self.used_prefixes.add(prefix)
            return c
        raise RuntimeError("canary code space exhausted")


def code_variants(code: str) -> list[str]:
    p, d, s = code.split("-")
    return [f"{p} {d} {s}", f"{p}{d}{s}", f"{p}-{d}", f"{p}{d}", f"{p}-{d}{s}", code.lower()]


def code_match_regex(code: str) -> str:
    """Case-insensitive regex tolerant to separator changes and a dropped suffix letter."""
    p, d, s = code.split("-")
    return rf"(?i)(?<![A-Za-z0-9]){p}[-\s]?{d}(?:[-\s]?{s})?(?![0-9])"


if __name__ == "__main__":  # quick self-check
    comp, ns = make_names(7, {"person": 90, "project": 36, "product": 20,
                              "team": 14, "vendor": 16, "site": 8})
    print(comp)
    for t, lst in ns.items():
        print(t, [n.canonical for n in lst[:4]])
    cf = CanaryCodeFactory(42)
    codes = [cf.code() for _ in range(80)]
    assert all(CODE_REGEX.fullmatch(c) for c in codes)
    assert all(re.search(code_match_regex(c), v) for c in codes for v in code_variants(c))
    print(codes[:5], code_variants(codes[0]))

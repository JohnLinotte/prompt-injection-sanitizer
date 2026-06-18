"""
Deterministic content sanitizer for prompt injection defense.

Provides regex-based detection of 20+ known injection patterns,
Unicode normalization, data-content tag escaping, risk scoring,
and a full sanitization pipeline. All functions are pure (input -> output)
except for an optional, user-injectable trace hook (see ``set_trace_hook``).

This is a standalone, dependency-free library: it runs deterministically and
is meant to process external/untrusted content BEFORE any LLM sees it.

Usage:
    from prompt_injection_sanitizer import (
        sanitize, SanitizationResult, DetectedPattern,
        escape_data_tags, wrap_in_data_tags,
        normalize_text, detect_injection_patterns,
        calculate_risk_score,
    )

    result = sanitize(text, source="email_body")
    if result.was_modified:
        print(f"Detected {len(result.detected_patterns)} patterns")
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

# Conditional import: prefer rapidfuzz for Damerau-Levenshtein (C++ speed),
# fall back to pure-Python implementation if not available.
try:
    from rapidfuzz.distance import DamerauLevenshtein as _RFDamerauLevenshtein

    def _damerau_levenshtein_distance(s1: str, s2: str) -> int:
        return _RFDamerauLevenshtein.distance(s1, s2)

except ImportError:

    def _damerau_levenshtein_distance(s1: str, s2: str) -> int:
        """Pure-Python optimal string alignment distance (restricted DL).

        Counts insertions, deletions, substitutions, and adjacent
        transpositions each as a single edit operation.
        """
        len_s1 = len(s1)
        len_s2 = len(s2)

        # Edge cases
        if len_s1 == 0:
            return len_s2
        if len_s2 == 0:
            return len_s1

        # Create matrix (len_s1+1) x (len_s2+1)
        d = [[0] * (len_s2 + 1) for _ in range(len_s1 + 1)]

        for i in range(len_s1 + 1):
            d[i][0] = i
        for j in range(len_s2 + 1):
            d[0][j] = j

        for i in range(1, len_s1 + 1):
            for j in range(1, len_s2 + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                d[i][j] = min(
                    d[i - 1][j] + 1,  # deletion
                    d[i][j - 1] + 1,  # insertion
                    d[i - 1][j - 1] + cost,  # substitution
                )
                # Transposition
                if (
                    i > 1
                    and j > 1
                    and s1[i - 1] == s2[j - 2]
                    and s1[i - 2] == s2[j - 1]
                ):
                    d[i][j] = min(d[i][j], d[i - 2][j - 2] + cost)

        return d[len_s1][len_s2]


# ===================================================================
# Optional trace hook
# ===================================================================

# Module-level trace hook. When set to a callable via set_trace_hook(), it is
# invoked once per sanitize() call that detects one or more patterns. It is
# optional: when None (the default), detection runs silently. The hook is for
# observability/audit only -- it must never affect the sanitization result.
_TRACE_HOOK = None


def set_trace_hook(hook) -> None:
    """Register (or clear) an optional trace hook for detected injections.

    When a hook is registered, :func:`sanitize` calls it once per invocation in
    which one or more patterns are detected. The hook is invoked with keyword
    arguments:

        hook(
            source=<str>,
            content_hash=<str>,           # SHA-256 of the original text
            detection_layer="sanitizer",
            patterns_detected=<list[str]>,  # detected pattern names
            risk_score=<float>,
            action_taken=<str>,           # "escaped" or "logged"
        )

    Exceptions raised by the hook are swallowed so a faulty hook can never break
    sanitization. Pass ``None`` to disable tracing again.

    Args:
        hook: A callable accepting the keyword arguments above, or None to
            disable tracing.
    """
    global _TRACE_HOOK
    _TRACE_HOOK = hook


# ===================================================================
# Dataclasses
# ===================================================================


@dataclass(frozen=True)
class DetectedPattern:
    """A single detected injection pattern."""

    pattern_name: str  # e.g., "ignore_instructions", "system_override"
    matched_text: str  # the actual matched text
    position: int  # character offset in original text
    risk_level: str  # "low", "medium", "high", "critical"


@dataclass(frozen=True)
class SanitizationResult:
    """Result of sanitizing a piece of content."""

    original_text: str
    sanitized_text: str
    detected_patterns: tuple[DetectedPattern, ...]
    risk_score: float  # 0.0 = clean, 1.0 = maximum risk
    content_hash: str  # SHA-256 of original for audit trail
    was_modified: bool
    source: str = ""  # e.g., "email_body", "web_fetch", "file_read"


# ===================================================================
# Module-level constants
# ===================================================================

MODIFICATION_THRESHOLD: float = 0.3

# Risk level weights for score calculation
_RISK_WEIGHTS: dict[str, float] = {
    "critical": 0.4,
    "high": 0.25,
    "medium": 0.15,
    "low": 0.05,
}

# Zero-width characters to strip during normalization
_ZERO_WIDTH_CHARS: str = "\u200b\u200c\u200d\ufeff\u00ad\u2060"

# Leet-speak substitution map (digit/symbol -> letter).
# Applied per-token only when a token mixes letters and digits/symbols,
# avoiding false positives on standalone numbers like "3.13" or "2026".
_LEET_MAP: dict[str, str] = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "8": "b",
    "9": "g",
    "@": "a",
    "$": "s",
    "!": "i",
    "(": "c",
    "|": "l",
}
_LEET_CHARS: frozenset[str] = frozenset(_LEET_MAP)

# Homoglyphs that survive NFKC: visually identical chars from other scripts.
# Keys are non-ASCII confusables, written as \uXXXX escapes to keep this
# source pure-ASCII; values are their Latin/ASCII look-alikes.
_HOMOGLYPH_MAP: dict[str, str] = {
    # Cyrillic -> Latin (Unicode confusables subset)
    "\u0430": "a",
    "\u0435": "e",
    "\u043e": "o",
    "\u0440": "p",
    "\u0441": "c",
    "\u0443": "y",  # visual
    "\u0445": "x",
    "\u0456": "i",
    "\u0458": "j",
    "\u0455": "s",
    "\u04bb": "h",
    "\u0501": "d",
    "\u0442": "t",  # Cyrillic TE
    "\u043d": "h",  # visual in sans-serif
    "\u043a": "k",
    "\u043c": "m",  # visual in some fonts
    "\u0432": "b",  # visual
    "\u0433": "r",  # visual in some fonts
    "\u043b": "n",  # visual in some sans-serif, rare
    "\u0410": "a",  # uppercase
    "\u0412": "b",
    "\u0415": "e",
    "\u041a": "k",
    "\u041c": "m",
    "\u041d": "h",
    "\u041e": "o",
    "\u0420": "p",
    "\u0421": "c",
    "\u0422": "t",
    "\u0425": "x",
    # Other scripts
    "\u0261": "g",  # Latin small script g
    "\u01c3": "!",  # Latin letter retroflex click
    # Greek
    "\u03bf": "o",  # Greek omicron
    "\u03b1": "a",  # visual
    "\u03c1": "p",  # visual
}
_HOMOGLYPH_CHARS: frozenset[str] = frozenset(_HOMOGLYPH_MAP)

_HAS_ALPHA_RE = re.compile(r"[a-zA-Z]")
_HAS_LEET_RE = re.compile(r"[0-9@$!|(]")

# ===================================================================
# Typoglycemia detection constants
# ===================================================================

# Critical keywords for fuzzy matching (typoglycemia defense).
CRITICAL_KEYWORDS: tuple[str, ...] = (
    "ignore",
    "instructions",
    "previous",
    "system",
    "prompt",
    "forget",
    "disregard",
    "override",
    "pretend",
    "bypass",
    "admin",
    "assistant",
    "execute",
    "command",
    "role",
    "jailbreak",
    # FR equivalents
    "ignorer",
    "oublier",
    "consignes",
    "precedentes",
    "contrainte",
    "restriction",
    "revele",
    "affiche",
)

# Critical multi-word phrases for fuzzy sequence matching.
# Each phrase is matched as a whole by comparing consecutive word windows.
CRITICAL_PHRASES: tuple[tuple[str, ...], ...] = (
    ("ignore", "previous", "instructions"),
    ("ignore", "all", "instructions"),
    ("disregard", "previous"),
    ("forget", "your", "instructions"),
    ("new", "role"),
    ("you", "are", "now"),
    ("system", "prompt"),
    ("override", "instructions"),
    # FR equivalents
    ("ignore", "instructions", "precedentes"),
    ("oublie", "les", "consignes"),
    ("prompt", "systeme"),
    ("libre", "contrainte"),
    ("sans", "restriction"),
)

# Pre-computed lookup sets for the first/last letter pre-filter.
# Typoglycemia preserves first+last letters, so we can skip tokens
# that don't share first OR last letter with any keyword.
_KEYWORD_FIRST_LETTERS: frozenset[str] = frozenset(kw[0] for kw in CRITICAL_KEYWORDS)
_KEYWORD_LAST_LETTERS: frozenset[str] = frozenset(kw[-1] for kw in CRITICAL_KEYWORDS)
# Index keywords by first letter for fast candidate filtering.
_KEYWORDS_BY_FIRST: dict[str, list[str]] = {}
for _kw in CRITICAL_KEYWORDS:
    _KEYWORDS_BY_FIRST.setdefault(_kw[0], []).append(_kw)
_KEYWORDS_BY_LAST: dict[str, list[str]] = {}
for _kw in CRITICAL_KEYWORDS:
    _KEYWORDS_BY_LAST.setdefault(_kw[-1], []).append(_kw)


# ===================================================================
# Compiled injection patterns (20+)
# ===================================================================

INJECTION_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    # (compiled_pattern, pattern_name, risk_level)
    # --- Direct instruction override attempts ---
    (
        re.compile(
            r"ignore\s+(all\s+)?(previous|prior|above|your)\s+instructions",
            re.IGNORECASE,
        ),
        "ignore_instructions",
        "critical",
    ),
    (
        re.compile(
            r"disregard\s+(all\s+)?(previous|prior|above|your)\s+"
            r"(instructions|rules|guidelines)",
            re.IGNORECASE,
        ),
        "disregard_instructions",
        "critical",
    ),
    (
        re.compile(
            r"forget\s+(everything|all)\s+(you|that)\s+(know|were|have)",
            re.IGNORECASE,
        ),
        "forget_instructions",
        "critical",
    ),
    # --- Role/identity hijacking ---
    (
        re.compile(
            r"you\s+are\s+now\s+(?:a|an|the)\s+",
            re.IGNORECASE,
        ),
        "identity_hijack",
        "high",
    ),
    (
        re.compile(
            r"act\s+as\s+(?:a|an|if)\s+",
            re.IGNORECASE,
        ),
        "role_override",
        "high",
    ),
    (
        re.compile(
            r"pretend\s+(to\s+be|you\s+are)\s+",
            re.IGNORECASE,
        ),
        "pretend_override",
        "high",
    ),
    (
        re.compile(
            r"your\s+new\s+(role|instructions?|task|purpose)\s+(is|are)",
            re.IGNORECASE,
        ),
        "new_role_assignment",
        "critical",
    ),
    # --- System prompt extraction ---
    (
        re.compile(
            r"(reveal|show|display|print|output|repeat)\s+(your\s+)?"
            r"(system\s+prompt|instructions|rules)",
            re.IGNORECASE,
        ),
        "system_prompt_extraction",
        "high",
    ),
    (
        re.compile(
            r"what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions|rules)",
            re.IGNORECASE,
        ),
        "system_prompt_query",
        "medium",
    ),
    # --- Encoding-based evasion ---
    (
        re.compile(
            r"base64\s*:\s*[A-Za-z0-9+/=]{20,}",
            re.IGNORECASE,
        ),
        "base64_payload",
        "medium",
    ),
    (
        re.compile(
            r"\\x[0-9a-fA-F]{2}(\\x[0-9a-fA-F]{2}){3,}",
            re.IGNORECASE,
        ),
        "hex_encoded_payload",
        "medium",
    ),
    # --- Developer/debug mode ---
    (
        re.compile(
            r"(enter|enable|switch\s+to|activate)\s+"
            r"(developer|debug|admin|god)\s+mode",
            re.IGNORECASE,
        ),
        "mode_switch",
        "critical",
    ),
    (
        re.compile(
            r"DAN\s+(mode|prompt|jailbreak)",
            re.IGNORECASE,
        ),
        "dan_jailbreak",
        "critical",
    ),
    # --- Data exfiltration ---
    (
        re.compile(
            r"(send|transmit|exfiltrate|post)\s+(to|data|this)\s+"
            r"(https?://|external)",
            re.IGNORECASE,
        ),
        "data_exfiltration",
        "critical",
    ),
    (
        re.compile(
            r"(include|embed|insert)\s+(in|into|as)\s+"
            r"(url|link|image|img\s+src)",
            re.IGNORECASE,
        ),
        "data_encoding_exfil",
        "high",
    ),
    # --- Delimiter manipulation ---
    (
        re.compile(
            r"</?(?:system|user|assistant|human|instructions?|prompt)>",
            re.IGNORECASE,
        ),
        "role_tag_injection",
        "high",
    ),
    (
        re.compile(
            r"\[\s*(?:SYSTEM|INST|SYS)\s*\]",
            re.IGNORECASE,
        ),
        "bracket_role_injection",
        "high",
    ),
    # --- Multi-step / indirect ---
    (
        re.compile(
            r"(first|step\s*1|before\s+anything)\s*[,:]\s*"
            r"(ignore|forget|disregard)",
            re.IGNORECASE,
        ),
        "multi_step_injection",
        "high",
    ),
    # --- Output manipulation ---
    (
        re.compile(
            r"(always|must|shall)\s+(respond|reply|answer|output)\s+with",
            re.IGNORECASE,
        ),
        "output_override",
        "medium",
    ),
    (
        re.compile(
            r"(do\s+not|never|don'?t)\s+(mention|reveal|tell|say|disclose)",
            re.IGNORECASE,
        ),
        "suppression_attempt",
        "medium",
    ),
    # --- French equivalents ---
    # FR instruction override
    (
        re.compile(
            r"ignor(?:e|ez)\s+(toutes?\s+)?(?:les\s+)?instructions?\s+"
            r"(?:pr[eé]c[eé]dentes?|ant[eé]rieures?|ci-dessus)",
            re.IGNORECASE,
        ),
        "fr_ignore_instructions",
        "critical",
    ),
    (
        re.compile(
            r"oubli(?:e|ez)\s+(?:toutes?\s+)?(?:les\s+)?(?:instructions?|r[eè]gles?|consignes?)",
            re.IGNORECASE,
        ),
        "fr_forget_instructions",
        "critical",
    ),
    (
        re.compile(
            r"ne\s+(?:tiens?|tenez)\s+(?:pas\s+)?compte\s+"
            r"(?:des?\s+)?(?:instructions?|r[eè]gles?|consignes?)",
            re.IGNORECASE,
        ),
        "fr_disregard_instructions",
        "critical",
    ),
    # FR role/identity hijacking
    (
        re.compile(
            r"tu\s+es\s+(?:maintenant|d[eé]sormais)\s+",
            re.IGNORECASE,
        ),
        "fr_identity_hijack",
        "high",
    ),
    (
        re.compile(
            r"(?:agis|comporte[- ]toi|fais)\s+comme\s+si\s+",
            re.IGNORECASE,
        ),
        "fr_role_override",
        "high",
    ),
    (
        re.compile(
            r"(?:tes?|vos?)\s+(?:nouvelles?\s+)?(?:instructions?|r[eè]gles?|r[oô]le|consignes?)"
            r"\s+(?:sont?|est|sera)",
            re.IGNORECASE,
        ),
        "fr_new_role_assignment",
        "critical",
    ),
    # FR system prompt extraction
    (
        re.compile(
            r"(?:r[eé]v[eè]le|montre|affiche|donne|r[eé]p[eè]te|imprime)"
            r"\s+(?:ton\s+|le\s+)?(?:prompt\s+syst[eè]me|instructions?\s+syst[eè]me|consignes?)",
            re.IGNORECASE,
        ),
        "fr_system_prompt_extraction",
        "high",
    ),
    (
        re.compile(
            r"(?:quelles?\s+sont|quel\s+est)\s+(?:ton\s+|tes\s+|le\s+)?"
            r"(?:prompt\s+syst[eè]me|instructions?\s+syst[eè]me|consignes?)",
            re.IGNORECASE,
        ),
        "fr_system_prompt_query",
        "medium",
    ),
    # FR output manipulation
    (
        re.compile(
            r"(?:r[eé]ponds?|r[eé]pondez)\s+(?:toujours|syst[eé]matiquement)\s+"
            r"(?:par|avec|en\s+disant)",
            re.IGNORECASE,
        ),
        "fr_output_override",
        "medium",
    ),
    # FR mode switch
    (
        re.compile(
            r"(?:active|passe\s+en|entre\s+en)\s+(?:mode\s+)?"
            r"(?:d[eé]veloppeur|debug|admin|dieu|god|sans\s+(?:limite|restriction|contrainte))",
            re.IGNORECASE,
        ),
        "fr_mode_switch",
        "critical",
    ),
    # FR suppression
    (
        re.compile(
            r"(?:ne\s+)?(?:mentionne|r[eé]v[eè]le|dis|divulgue)"
            r"\s+(?:pas|jamais|surtout\s+pas)",
            re.IGNORECASE,
        ),
        "fr_suppression_attempt",
        "medium",
    ),
    # FR no restrictions / free from constraints
    (
        re.compile(
            r"(?:libre|libr[eé]r[eé]|affranchi|d[eé]gag[eé])"
            r"\s+(?:de\s+)?(?:toute|toutes)\s+(?:contrainte|restriction|r[eè]gle|limite)",
            re.IGNORECASE,
        ),
        "fr_unrestricted",
        "high",
    ),
    # FR exfiltration (handle French articles: l', la, le, les)
    (
        re.compile(
            r"(?:envoie|transmets?|transf[eè]re)"
            r"(?:\s+(?:tout|toute|tous|toutes))?"
            r"(?:\s+(?:l[ea']?s?|la|le|l'))?\s*"
            r"(?:historique|conversation|donn[eé]es?|m[eé]moire|contexte)",
            re.IGNORECASE,
        ),
        "fr_data_exfiltration",
        "critical",
    ),
    # FR command execution
    (
        re.compile(
            r"(?:ex[eé]cute|lance|fais)\s+(?:la\s+)?(?:commande|instruction)\s+suivante",
            re.IGNORECASE,
        ),
        "fr_command_execution",
        "high",
    ),
    # FR indirect injection ("quand on te demande X, reponds Y")
    (
        re.compile(
            r"(?:quand|lorsqu|si)\s+(?:on\s+te|quelqu'?un)\s+"
            r"(?:demand|interrog|parl|pos)",
            re.IGNORECASE,
        ),
        "fr_indirect_injection",
        "medium",
    ),
    # FR "aucune restriction/contrainte"
    (
        re.compile(
            r"(?:aucune?|sans|plus\s+de)\s+"
            r"(?:restriction|contrainte|limite|r[eè]gle|garde-?fou)",
            re.IGNORECASE,
        ),
        "fr_no_restrictions",
        "high",
    ),
    # --- Additional EN gap coverage ---
    # EN: broader "no restrictions" pattern
    (
        re.compile(
            r"(?:you\s+have|with)\s+no\s+(?:restrictions?|limitations?|constraints?|rules?|guardrails?)",
            re.IGNORECASE,
        ),
        "en_unrestricted",
        "high",
    ),
    # EN: broader exfiltration (verb ... url anywhere)
    (
        re.compile(
            r"(?:send|forward|post|upload|transmit)\s+.*?"
            r"(?:conversation|history|memory|context|data)\s+.*?https?://",
            re.IGNORECASE,
        ),
        "en_data_exfil_broad",
        "critical",
    ),
    # EN: elevated privileges / debug mode without "enter/enable" prefix
    (
        re.compile(
            r"(?:operating|running|working)\s+in\s+"
            r"(?:debug|admin|elevated|root|privileged|unrestricted)\s+(?:mode|privileges?)",
            re.IGNORECASE,
        ),
        "en_elevated_mode",
        "high",
    ),
    # EN: deferred injection ("remember this for later", "when X do Y")
    (
        re.compile(
            r"(?:remember|store|save)\s+(?:this|the\s+following)\s+(?:for\s+later|instruction)",
            re.IGNORECASE,
        ),
        "en_deferred_injection",
        "medium",
    ),
    # EN: command execution instruction
    (
        re.compile(
            r"(?:execute|run)\s+(?:the\s+following|this)\s+(?:command|script|code)",
            re.IGNORECASE,
        ),
        "en_command_execution",
        "high",
    ),
    # EN: broader jailbreak -- "pretend you have no X"
    (
        re.compile(
            r"pretend\s+(?:you\s+)?(?:have|had)\s+no\s+"
            r"(?:restrictions?|limitations?|constraints?|rules?|guardrails?|boundaries)",
            re.IGNORECASE,
        ),
        "en_pretend_unrestricted",
        "critical",
    ),
]

# Pre-filter: at least one of these substrings must appear in the
# normalized text for any INJECTION_PATTERN to match.  Checked via
# str.__contains__ (CPython Boyer-Moore) -- ~0.3ms for 100KB vs ~100ms
# for the full 42-pattern scan.  Covers every regex above.
_INJECTION_SIGNAL_WORDS: frozenset[str] = frozenset(
    {
        # EN anchors
        "ignore",
        "disregard",
        "forget",
        "pretend",
        "you are now",
        "act as ",
        "your new ",
        "reveal",
        "display",
        "system prompt",
        "system ",
        "instructions",
        "base64",
        "\\x",
        "mode",
        "dan ",
        "send ",
        "transmit",
        "exfiltrat",
        "include ",
        "embed ",
        "respond with",
        "reply with",
        "answer with",
        "output with",
        "not mention",
        "never reveal",
        "never tell",
        "first,",
        "first:",
        "step 1",
        "before anything",
        "<system",
        "</system",
        "<user",
        "</user",
        "<assistant",
        "<human",
        "<instruct",
        "<prompt",
        "</prompt",
        "[system",
        "[inst",
        "[sys",
        "no restriction",
        "no limitation",
        "no constraint",
        "no rule",
        "no guardrail",
        "operating in",
        "running in",
        "working in",
        "remember this",
        "store this",
        "save this",
        "execute the",
        "execute this",
        "run the",
        "run this",
        # Prefilter completeness: these anchors had been missing, letting real
        # patterns slip past the signal-word gate (e.g. "post this https://..."
        # matched data_exfiltration's regex but no anchor fired -> silent miss).
        # Every alternation branch of every INJECTION_PATTERN MUST have a covering
        # substring here; pinned by the prefilter-coverage test suite.
        "post ",
        "insert ",
        "forward",
        "upload",  # exfil / encoding verb branches
        "disclose",
        "not tell",
        "don't tell",
        "not say",
        "never say",
        "don't say",
        "never mention",
        "don't mention",
        "not reveal",
        "don't reveal",  # suppression verbs
        "the following",  # en_deferred_injection "the following" branch
        "repeat",
        "output ",
        "rules",  # system_prompt_extraction verb/object branches
        # FR anchors
        "ignor",
        "oubli",
        "consigne",
        "tiens pas compte",
        "tenez pas compte",
        "tu es maintenant",
        "tu es d\u00e9sormais",
        "tu es desormais",
        "agis comme",
        "comporte",
        "fais comme",
        "r\u00e9v\u00e8le",
        "revele",
        "montre ",
        "affiche ",
        "donne ",
        "imprime",
        "r\u00e9p\u00e8te",
        "repete",
        "prompt syst\u00e8",
        "prompt syste",
        "instructions syst\u00e8",
        "instructions syste",
        "r\u00e9ponds",
        "reponds",
        "active ",
        "passe en",
        "entre en",
        "mentionne",
        "divulgue",
        "libre ",
        "lib\u00e9r\u00e9",
        "libere",
        "affranchi",
        "contrainte",
        "restriction",
        "envoie",
        "transmets",
        "transf\u00e8re",
        "transfere",
        "ex\u00e9cute",
        "execute",
        "lance ",
        "commande ",
        "quand on te",
        "lorsqu",
        "quelqu",
        "aucune",
        "sans restriction",
        "sans contrainte",
        "sans limite",
        "plus de restriction",
        "plus de contrainte",
    }
)


# ===================================================================
# Data-content tag patterns
# ===================================================================

_DATA_TAG_OPEN: re.Pattern[str] = re.compile(r"<data-content\b", re.IGNORECASE)
_DATA_TAG_CLOSE: re.Pattern[str] = re.compile(r"</data-content>", re.IGNORECASE)


# ===================================================================
# Core functions
# ===================================================================


def _deleet_token(token: str) -> str:
    """Replace leet-speak chars in a single token that mixes alpha+digits."""
    if not (_HAS_ALPHA_RE.search(token) and _HAS_LEET_RE.search(token)):
        return token
    return "".join(_LEET_MAP.get(ch, ch) for ch in token)


def normalize_text(text: str) -> str:
    """Normalize text for consistent pattern matching.

    Applies NFKC normalization, homoglyph replacement (Cyrillic look-alikes
    that survive NFKC), zero-width character removal, and casefolding.

    Leet-speak de-substitution is NOT applied here because it would mangle
    legitimate digit sequences (e.g. "base64" -> "baseba"). Instead,
    leet-speak detection runs as a separate pass via detect_leet_injection().

    Args:
        text: Raw text to normalize.

    Returns:
        Normalized text suitable for pattern matching.
    """
    # NFKC normalization
    normalized = unicodedata.normalize("NFKC", text)

    # Remove zero-width characters
    for char in _ZERO_WIDTH_CHARS:
        normalized = normalized.replace(char, "")

    # Homoglyph replacement (Cyrillic/Latin look-alikes that survive NFKC)
    if _HOMOGLYPH_CHARS.intersection(normalized):
        normalized = "".join(_HOMOGLYPH_MAP.get(ch, ch) for ch in normalized)

    # Casefold for case-insensitive matching
    return normalized.casefold()


def _deleet_text(text: str) -> str:
    """Apply leet-speak de-substitution to already-normalized text.

    Only transforms tokens that mix alpha and leet chars, leaving
    standalone numbers ("3.13", "2026", "192.168.1.1") untouched.
    """
    if not _HAS_LEET_RE.search(text):
        return text
    parts = []
    i = 0
    for m in re.finditer(r"\S+", text):
        parts.append(text[i : m.start()])
        parts.append(_deleet_token(m.group()))
        i = m.end()
    parts.append(text[i:])
    return "".join(parts)


def detect_injection_patterns(text: str) -> list[DetectedPattern]:
    """Detect known prompt injection patterns via regex.

    Normalizes text first (NFKC + zero-width removal + casefold), then
    matches all INJECTION_PATTERNS. Returns matches sorted by position.

    Note: Positions are reported from the normalized text since the
    original text may have different character offsets due to Unicode
    normalization and zero-width character removal.

    Args:
        text: Content to scan for injection patterns.

    Returns:
        List of DetectedPattern instances sorted by position.
    """
    normalized = normalize_text(text)

    # Fast pre-filter: skip expensive 42-pattern scan when no signal word
    # appears in the text.  Benign content (99%+ of traffic) exits here.
    if not any(w in normalized for w in _INJECTION_SIGNAL_WORDS):
        return []

    detected: list[DetectedPattern] = []

    for pattern, name, risk_level in INJECTION_PATTERNS:
        for match in pattern.finditer(normalized):
            detected.append(
                DetectedPattern(
                    pattern_name=name,
                    matched_text=match.group(),
                    position=match.start(),
                    risk_level=risk_level,
                )
            )

    # Sort by position
    detected.sort(key=lambda d: d.position)
    return detected


def detect_typoglycemia(text: str) -> list[DetectedPattern]:
    """Detect typoglycemia-obfuscated injection keywords and phrases.

    Typoglycemia exploits the fact that humans (and LLMs) can read words
    with scrambled middle letters as long as the first and last letters
    are correct. Attackers use this to bypass regex filters:
    e.g. "ignroe prevoius instrctions" -> "ignore previous instructions".

    Uses Damerau-Levenshtein distance (transpositions count as 1 edit)
    with a first/last letter pre-filter for performance.

    Two detection passes:
        1. Single-keyword fuzzy match: flags individual scrambled keywords.
        2. Phrase window match: looks for consecutive fuzzy-matched words
           that form a known critical phrase (higher confidence).

    Args:
        text: Raw text to scan (will be normalized internally).

    Returns:
        List of DetectedPattern instances for fuzzy matches.
    """
    normalized = normalize_text(text)
    # Tokenize by whitespace and strip punctuation from edges
    raw_tokens = normalized.split()
    if not raw_tokens:
        return []

    # Performance cap: for very large texts, only scan the first and last
    # MAX_SCAN_TOKENS tokens. Injection payloads are typically at the start
    # of injected content or at document boundaries, not buried in the middle
    # of 10KB+ of legitimate prose.
    _MAX_SCAN_TOKENS = 200
    if len(raw_tokens) > _MAX_SCAN_TOKENS * 2:
        # Take first N and last N tokens, preserving position info
        head = raw_tokens[:_MAX_SCAN_TOKENS]
        tail = raw_tokens[-_MAX_SCAN_TOKENS:]
        # Calculate the character offset where tail starts
        head_char_len = sum(
            len(t) + 1 for t in raw_tokens[: len(raw_tokens) - _MAX_SCAN_TOKENS]
        )
        scan_segments = [(head, 0), (tail, head_char_len)]
    else:
        scan_segments = [(raw_tokens, 0)]

    # Clean tokens: strip common punctuation
    tokens: list[tuple[str, int]] = []  # (clean_token, char_position)
    for segment, base_pos in scan_segments:
        pos = base_pos
        for raw in segment:
            clean = raw.strip(".,!?;:\"'()[]{}/<>")
            tokens.append((clean, pos))
            pos += len(raw) + 1  # +1 for the space

    detected: list[DetectedPattern] = []
    # Track which tokens matched which keywords for phrase detection
    token_keyword_map: list[str | None] = []

    # ---- Pass 1: Single keyword fuzzy matching ----
    for clean, tok_pos in tokens:
        matched_kw = None

        # Skip empty tokens
        if not clean:
            token_keyword_map.append(None)
            continue

        # Skip tokens too short for fuzzy matching
        if len(clean) < 4:
            # Exact match only for short tokens
            if clean in CRITICAL_KEYWORDS:
                matched_kw = clean
            token_keyword_map.append(matched_kw)
            continue

        # Length range guard: all critical keywords are 4-12 chars.
        # With 30% distance tolerance, a token matching a 12-char keyword
        # can be at most 12+3=15 chars. Skip tokens outside plausible range.
        if len(clean) > 15:
            token_keyword_map.append(None)
            continue

        # Pre-filter: typoglycemia preserves first AND last letters.
        # Token must share first letter with at least one keyword AND
        # last letter with at least one keyword to be a candidate.
        first = clean[0]
        last = clean[-1]
        candidates_first = _KEYWORDS_BY_FIRST.get(first, [])
        candidates_last = set(_KEYWORDS_BY_LAST.get(last, []))

        if not candidates_first or not candidates_last:
            token_keyword_map.append(None)
            continue

        # Intersect: keywords that match BOTH first and last letter
        candidates = [kw for kw in candidates_first if kw in candidates_last]
        if not candidates:
            token_keyword_map.append(None)
            continue

        for kw in candidates:
            max_dist = max(1, int(len(kw) * 0.3))
            # Typoglycemia is an ANAGRAM (same letters, middle reordered): a
            # genuine scramble shares the keyword's letter multiset, and the only
            # legitimate deviation is a single dropped/added letter. Reject when
            # the multiset symmetric difference exceeds 1 -- that marks a
            # DIFFERENT real word that merely lands within edit distance
            # (e.g. FR "prend" vs "pretend", or "ignare" vs "ignore"), not a
            # scramble. This subsumes the old length guard (a length gap >= 2
            # implies a multiset diff >= 2) and additionally rejects same-length
            # substitutions, which are leetspeak/homoglyph attacks -- the job of
            # normalize_text, not of typoglycemia detection.
            c_tok, c_kw = Counter(clean), Counter(kw)
            if sum((c_tok - c_kw).values()) + sum((c_kw - c_tok).values()) > 1:
                continue

            # Skip exact matches (already caught by regex detection)
            if clean == kw:
                matched_kw = kw
                break

            dist = _damerau_levenshtein_distance(clean, kw)
            if dist <= max_dist and dist > 0:
                matched_kw = kw
                # Record as detection (single keyword = "high" risk)
                detected.append(
                    DetectedPattern(
                        pattern_name=f"typoglycemia:{kw}",
                        matched_text=clean,
                        position=tok_pos,
                        risk_level="high",
                    )
                )
                break

        token_keyword_map.append(matched_kw)

    # ---- Pass 2: Phrase window matching ----
    # Look for consecutive tokens that match a critical phrase.
    # This catches "ignroe prevoius instrctions" as a phrase, even if
    # individual words wouldn't be alarming alone.
    for phrase in CRITICAL_PHRASES:
        phrase_len = len(phrase)
        for i in range(len(token_keyword_map) - phrase_len + 1):
            window = token_keyword_map[i : i + phrase_len]
            if all(w is not None and w == phrase[j] for j, w in enumerate(window)):
                # Full phrase matched via fuzzy tokens
                phrase_text = " ".join(tokens[i + j][0] for j in range(phrase_len))
                phrase_name = "_".join(phrase)
                # Avoid duplicate if exact same position already detected
                start_pos = tokens[i][1]
                already = any(
                    d.position == start_pos
                    and d.pattern_name == f"typoglycemia_phrase:{phrase_name}"
                    for d in detected
                )
                if not already:
                    detected.append(
                        DetectedPattern(
                            pattern_name=f"typoglycemia_phrase:{phrase_name}",
                            matched_text=phrase_text,
                            position=start_pos,
                            risk_level="high",
                        )
                    )

    # Sort by position
    detected.sort(key=lambda d: d.position)
    return detected


_RISK_PROMOTE: dict[str, str] = {
    "low": "medium",
    "medium": "high",
    "high": "critical",
    "critical": "critical",
}


def detect_leet_injection(text: str) -> list[DetectedPattern]:
    """Detect injection patterns hidden via leet-speak substitution.

    Creates a de-leeted copy of the normalized text and runs all regex
    patterns against it. Only returns NEW matches that would not have
    been found by detect_injection_patterns() on the original text.

    Matches found only through de-leeting are promoted one risk level
    (medium->high, high->critical) because the obfuscation itself is
    evidence of adversarial intent.

    This is a separate pass (not folded into normalize_text) because
    de-leeting mangles legitimate digit sequences like "base64".
    """
    normalized = normalize_text(text)
    deleeted = _deleet_text(normalized)

    if deleeted == normalized:
        return []

    # Find patterns in the de-leeted text
    detected: list[DetectedPattern] = []
    for pattern, name, risk_level in INJECTION_PATTERNS:
        for match in pattern.finditer(deleeted):
            detected.append(
                DetectedPattern(
                    pattern_name=f"leet:{name}",
                    matched_text=match.group(),
                    position=match.start(),
                    risk_level=_RISK_PROMOTE.get(risk_level, risk_level),
                )
            )

    if not detected:
        return []

    # Subtract patterns already found in the non-deleeted text to avoid
    # double-counting (same base pattern, same position = same finding).
    original_detected = set()
    for pattern, name, _ in INJECTION_PATTERNS:
        for match in pattern.finditer(normalized):
            original_detected.add((name, match.start()))

    novel = [
        d
        for d in detected
        if (d.pattern_name.removeprefix("leet:"), d.position) not in original_detected
    ]
    novel.sort(key=lambda d: d.position)
    return novel


def escape_data_tags(text: str) -> str:
    """Escape <data-content> tags within content to prevent boundary breaking.

    Does NOT escape other HTML/XML -- only the specific tags used for
    the data boundary produced by wrap_in_data_tags(). This preserves
    legitimate HTML in emails, documents, etc.

    Args:
        text: Raw content that may contain data-content tag strings.

    Returns:
        Text with data-content tags escaped via HTML entities.
    """
    # Escape closing tags first (more dangerous -- breaks containment)
    result = _DATA_TAG_CLOSE.sub("&lt;/data-content&gt;", text)
    # Escape opening tags
    result = _DATA_TAG_OPEN.sub("&lt;data-content", result)
    return result


def wrap_in_data_tags(text: str, source: str) -> str:
    """Wrap content in data-content tags after escaping.

    Calls escape_data_tags on the text, then wraps in a tagged boundary
    with source attribution.

    Args:
        text: Content to wrap (will be escaped first).
        source: Source identifier (e.g., "email_body", "web_fetch").

    Returns:
        Tagged content string.
    """
    escaped = escape_data_tags(text)
    safe_source = html.escape(source)
    return f'<data-content source="{safe_source}">\n{escaped}\n</data-content>'


# Inverse of the ``[SANITIZED: <pattern>]...[/SANITIZED]`` annotation applied by
# ``sanitize`` when risk >= MODIFICATION_THRESHOLD. ``pattern_name`` never
# contains ']' (see DetectedPattern.pattern_name), so a non-greedy inner capture
# is unambiguous; re.DOTALL lets a marked span run across newlines.
_SANITIZED_MARKER_RE = re.compile(r"\[SANITIZED:[^\]]*\](.*?)\[/SANITIZED\]", re.DOTALL)


def strip_sanitization_markers(text: str) -> str:
    """Remove ``[SANITIZED: ...]...[/SANITIZED]`` markers, keeping inner content.

    Deterministic inverse of the marker-annotation step in :func:`sanitize`.
    ``sanitize`` never *deletes* content -- it wraps the matched span in markers
    -- so this peels the markers back off and returns the wrapped content
    verbatim. Idempotent: text with no markers is returned unchanged, and a
    second pass strips nothing further.

    Note (lossy round-trip): when ``sanitize`` fires it annotates the
    *normalized* text (NFKC + casefold + zero-width strip via
    :func:`normalize_text`), so for fired input ``x``::

        strip_sanitization_markers(sanitize(x).sanitized_text)
            == escape_data_tags(normalize_text(x))

    Markers are recovered exactly, but the upstream normalization is *not*
    reversed. For input that does not fire, the result equals
    ``escape_data_tags(x)``.

    Args:
        text: Text that may contain sanitization markers.

    Returns:
        Text with all sanitization markers removed, inner content preserved.
    """
    if "[SANITIZED:" not in text:
        return text
    return _SANITIZED_MARKER_RE.sub(r"\1", text)


def calculate_risk_score(patterns: list[DetectedPattern]) -> float:
    """Calculate aggregate risk score from detected patterns.

    Weights by risk level: critical=0.4, high=0.25, medium=0.15, low=0.05.
    Sums weights and clamps to [0.0, 1.0].

    Args:
        patterns: List of detected injection patterns.

    Returns:
        Risk score between 0.0 (clean) and 1.0 (maximum risk).
    """
    if not patterns:
        return 0.0

    total = sum(_RISK_WEIGHTS.get(p.risk_level, 0.05) for p in patterns)
    return min(total, 1.0)


def sanitize(text: str, source: str = "") -> SanitizationResult:
    """Full sanitization pipeline for external content.

    Steps:
        1. Compute content hash (SHA-256 of original).
        2. Detect injection patterns (regex) -- runs on all input lengths.
        3. Short-circuit only for short text with no detections/evasion signal.
        4. Calculate risk score.
        5. If risk >= MODIFICATION_THRESHOLD: annotate matched text with
           [SANITIZED: pattern_name] markers. Never deletes content.
        6. Escape data-content tags in the result.
        7. If patterns detected and a trace hook is registered, call it.
        8. Return SanitizationResult.

    Args:
        text: External content to sanitize.
        source: Source identifier for audit trail.

    Returns:
        SanitizationResult with sanitized text, detected patterns,
        risk score, and modification status.
    """
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    # Regex injection detection always runs, including on short text.
    # ``detect_injection_patterns`` has its own cheap signal-word pre-filter
    # (benign content exits in O(1)), so this is safe for the hot path while
    # ensuring canonical short payloads such as
    # "ignore all previous instructions" (len 32) are never silently skipped.
    detected = detect_injection_patterns(text)

    # Typoglycemia detection runs unconditionally for non-trivial input
    # because scrambled injection payloads are often short (< 50 chars).
    # The fuzzy matcher is fast enough for short texts (< 1ms).
    typo_detected = detect_typoglycemia(text) if len(text) >= 10 else []

    # Check if normalization would change the text (leet-speak or homoglyphs).
    # If so, the leet-aware pass must still run even on short input.
    _has_evasion_chars = False
    if len(text) < 50:
        _has_evasion_chars = bool(_HOMOGLYPH_CHARS.intersection(text)) or any(
            _HAS_ALPHA_RE.search(tok) and _HAS_LEET_RE.search(tok)
            for tok in text.split()
        )

    # Short-circuit for very short, clearly-benign text (performance): skip
    # the remaining leet/typoglycemia merge work only when nothing has been
    # detected and no evasion signal is present. Regex detection above has
    # already had its say, so a real injection in short text still surfaces.
    if (
        len(text) < 50
        and not detected
        and not typo_detected
        and not _has_evasion_chars
    ):
        escaped = escape_data_tags(text)
        return SanitizationResult(
            original_text=text,
            sanitized_text=escaped,
            detected_patterns=(),
            risk_score=0.0,
            content_hash=content_hash,
            was_modified=(escaped != text),
            source=source,
        )

    # Leet-speak detection: run regex on de-leeted copy, return novel matches
    leet_detected = (
        detect_leet_injection(text) if _has_evasion_chars or len(text) >= 50 else []
    )
    detected.extend(leet_detected)

    # Merge typoglycemia results (already computed above)
    detected.extend(typo_detected)

    risk_score = calculate_risk_score(detected)

    # Apply sanitization if risk is above threshold
    sanitized = text
    was_modified = False

    if risk_score >= MODIFICATION_THRESHOLD and detected:
        # Annotate detected patterns in the text.
        # We work on the normalized text to find positions, then apply
        # markers. To avoid offset drift, process matches in reverse
        # position order so earlier positions remain valid.
        normalized = normalize_text(text)
        sanitized = normalized

        # Sort by position descending for safe in-place replacement
        sorted_patterns = sorted(detected, key=lambda d: d.position, reverse=True)

        for dp in sorted_patterns:
            start = dp.position
            end = start + len(dp.matched_text)
            original_match = sanitized[start:end]
            marker = f"[SANITIZED: {dp.pattern_name}]{original_match}[/SANITIZED]"
            sanitized = sanitized[:start] + marker + sanitized[end:]

        was_modified = True

    # Escape data-content tags in the result
    sanitized = escape_data_tags(sanitized)
    if sanitized != text and not was_modified:
        was_modified = True

    # Optional trace logging for detected patterns. The hook is user-injectable
    # via set_trace_hook(); when none is registered we skip silently. A faulty
    # hook must never break sanitization.
    if detected and _TRACE_HOOK is not None:
        try:
            _TRACE_HOOK(
                source=source,
                content_hash=content_hash,
                detection_layer="sanitizer",
                patterns_detected=[d.pattern_name for d in detected],
                risk_score=risk_score,
                action_taken="escaped" if was_modified else "logged",
            )
        except Exception:
            # Catch-all is intentional here: an arbitrary user-provided hook
            # must never be able to break the sanitization result. This is not a
            # bare except -- it explicitly targets Exception, so BaseException
            # control-flow exits (e.g. KeyboardInterrupt) still propagate.
            pass

    return SanitizationResult(
        original_text=text,
        sanitized_text=sanitized,
        detected_patterns=tuple(detected),
        risk_score=risk_score,
        content_hash=content_hash,
        was_modified=was_modified,
        source=source,
    )

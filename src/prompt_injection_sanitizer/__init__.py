"""prompt-injection-sanitizer: deterministic prompt-injection defense.

A dependency-free, deterministic sanitizer for untrusted text destined for an
LLM. It detects 20+ known injection patterns by regex, normalizes Unicode
(NFKC, homoglyph folding, zero-width stripping), defeats typoglycemia and
leetspeak obfuscation, escapes data-content boundary tags, and scores risk.

Public API re-exported here:

    sanitize, SanitizationResult, DetectedPattern,
    escape_data_tags, wrap_in_data_tags, normalize_text,
    detect_injection_patterns, detect_typoglycemia, detect_leet_injection,
    strip_sanitization_markers, calculate_risk_score, set_trace_hook
"""

from ._core import (
    DetectedPattern,
    SanitizationResult,
    calculate_risk_score,
    detect_injection_patterns,
    detect_leet_injection,
    detect_typoglycemia,
    escape_data_tags,
    normalize_text,
    sanitize,
    set_trace_hook,
    strip_sanitization_markers,
    wrap_in_data_tags,
)

__version__ = "0.1.0"

__all__ = [
    "sanitize",
    "SanitizationResult",
    "DetectedPattern",
    "escape_data_tags",
    "wrap_in_data_tags",
    "normalize_text",
    "detect_injection_patterns",
    "detect_typoglycemia",
    "detect_leet_injection",
    "strip_sanitization_markers",
    "calculate_risk_score",
    "set_trace_hook",
]

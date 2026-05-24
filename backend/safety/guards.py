import re

from backend.safety.config import safety_config

_PII_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"),
    "phone": re.compile(r"\b(\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}

_TOXIC_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(hate|kill|destroy|attack)\s+(all\s+)?(people|group|race|religion)\b", re.I),
]


class GuardRailsValidator:
    """Validates and optionally redacts LLM output using configurable safety rules."""

    def validate(self, text: str) -> tuple[str, list[str]]:
        """Return (validated_text, violations). When disabled, acts as a passthrough."""
        if not safety_config.enabled:
            return text, []

        validated = text
        violations: list[str] = []

        if safety_config.redact_pii:
            validated, pii_violations = self._redact_pii(validated)
            violations.extend(pii_violations)

        toxic_violations = self._check_toxicity(validated)
        violations.extend(toxic_violations)

        allowlist = safety_config.topic_allowlist
        if allowlist:
            off_topic = self._check_topic(validated, allowlist)
            violations.extend(off_topic)

        return validated, violations

    def _redact_pii(self, text: str) -> tuple[str, list[str]]:
        """Replace detected PII with redaction placeholders."""
        violations = []
        for label, pattern in _PII_PATTERNS.items():
            if pattern.search(text):
                text = pattern.sub(f"[REDACTED_{label.upper()}]", text)
                violations.append(f"PII detected and redacted: {label}")
        return text, violations

    def _check_toxicity(self, text: str) -> list[str]:
        """Return violation messages for any matched toxic language patterns."""
        return [
            f"Toxic language pattern detected: '{m.group()}'"
            for pattern in _TOXIC_PATTERNS
            for m in pattern.finditer(text)
        ]

    def _check_topic(self, text: str, allowlist: list[str]) -> list[str]:
        """Flag content that does not mention any topic in the allowlist."""
        text_lower = text.lower()
        if not any(topic.lower() in text_lower for topic in allowlist):
            return [f"Content may be off-topic. Allowed topics: {', '.join(allowlist)}"]
        return []

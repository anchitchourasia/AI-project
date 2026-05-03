"""
Safety Guard — pure local computation, no LLM, no network.
Must complete in well under 10 ms for any input.

Logic: two-pass check
  Pass 1 — BLOCK patterns  (harmful intent keywords)
  Pass 2 — PASS patterns   (educational/definitional signals)

If both pass-1 and pass-2 fire → education wins → NOT blocked.
If only pass-1 fires           → BLOCKED.
"""
from __future__ import annotations
import re
from src.models import SafetyVerdict

_CATEGORIES: dict[str, dict] = {
    "insider_trading": {
        "block_patterns": [
            r"\b(i\s+work\s+at|my\s+friend\s+at|tip\s+from|inside\s+information"
            r"|confidential\s+(merger|acquisition|deal)"
            r"|unannounced\s+(acquisition|earnings|deal)"
            r"|trade\s+on\s+(this|confidential|non.?public)"
            r"|earnings\s+before\s+(tomorrow|the\s+announcement)"
            r"|load\s+up\s+before\s+the\s+call"
            r"|need\s+to\s+know\s+the\s+earnings)\b",
        ],
        "pass_patterns": [
            r"\b(what\s+is|explain|definition|penalty|penalties|how\s+does|how\s+do"
            r"|difference\s+between|investigate|fca|sec|disclosure\s+rules"
            r"|material\s+non.?public|front.?running|regulations?|illegal|law)\b",
        ],
        "message": (
            "I can't assist with trading on material non-public or confidential information. "
            "Insider trading is a serious criminal offence under securities law worldwide. "
            "Please consult a compliance officer or legal counsel."
        ),
    },
    "market_manipulation": {
        "block_patterns": [
            r"\b(pump\s+(up|the\s+price)|pump.and.dump"
            r"|coordinated\s+buying|move\s+(this\s+stock|the\s+price)"
            r"|wash\s+trad(e|ing)\s+(between|to\s+create)"
            r"|design\s+a\s+coordinated|create\s+fake\s+volume"
            r"|spoof(ing)?\s+(orders?|bids?))\b",
        ],
        "pass_patterns": [
            r"\b(what\s+is|explain|definition|how\s+do|how\s+does|detect"
            r"|illegal|regulators?|sec|fca|legal\?"
            r"|pump.and.dump\s+(group|scheme)"
            r"|spoofing\s+and\s+layering|wash\s+trading\s+and\s+why)\b",
        ],
        "message": (
            "I'm unable to assist with market manipulation strategies. "
            "Coordinated buying schemes, pump-and-dump operations, and wash trading are illegal "
            "under market abuse regulations globally and carry severe criminal penalties."
        ),
    },
    "money_laundering": {
        "block_patterns": [
            r"\b(move\s+.{0,30}\s+without\s+reporting|structure\s+deposits"
            r"|avoid\s+(the\s+)?10[k,]?\s+reporting|layer\s+.{0,30}\s+obscure"
            r"|hide\s+(trading\s+profits?|funds?|money)"
            r"|without\s+it\s+being\s+traced)\b",
        ],
        "pass_patterns": [
            r"\b(what\s+(are|is)|explain|aml|reporting\s+requirements"
            r"|structuring\s+in|three\s+stages"
            r"|placement.{0,10}layering.{0,10}integration"
            r"|definition|compliance|regulation|illegal"
            r"|how\s+do\s+brokerages?)\b",
        ],
        "message": (
            "I can't help with concealing the source of funds, structuring transactions, "
            "or evading financial reporting obligations. These activities constitute money "
            "laundering and are prosecuted under criminal law in every major jurisdiction."
        ),
    },
    "guaranteed_returns": {
        "block_patterns": [
            r"\b(guarantee\s+(me|my)|guaranteed\s+return|promise\s+me"
            r"|100%\s+certain\s+to\s+go\s+up|foolproof\s+way\s+to\s+make"
            r"|will\s+double\s+in\s+a\s+year"
            r"|certain\s+to\s+(go\s+up|profit)"
            r"|stock\s+that.{0,10}100%)\b",
        ],
        "pass_patterns": [
            r"\b(legitimate|advertise|historical\s+(average|return)|red\s+flag"
            r"|ponzi|fraud|why\s+are|what.{0,10}guaranteed.{0,10}claim"
            r"|what.{0,10}s\s+the\s+historical|explain)\b",
        ],
        "message": (
            "No investment returns can be guaranteed. "
            "Any claim of guaranteed returns on equity investments is a hallmark of fraud "
            "and may violate financial promotion rules. Please be cautious of such claims."
        ),
    },
    "reckless_advice": {
        "block_patterns": [
            r"\b(put\s+all\s+my\s+(retirement\s+savings?|life\s+savings?|emergency\s+fund)"
            r"\s+in(to)?\s+(crypto|options?|single\s+stock)"
            r"|take\s+a\s+margin\s+loan\s+to\s+buy\s+more"
            r"|mortgage\s+my\s+house\s+for"
            r"|entire\s+emergency\s+fund\s+into\s+options?"
            r"|tell\s+me\s+to\s+take\s+a\s+margin"
            r"|which\s+stock\s+to\s+mortgage)\b",
        ],
        "pass_patterns": [
            r"\b(what\s+factors?|risks?\s+of|how\s+should"
            r"|emergency\s+fund\s+be\s+allocated|leveraged\s+investing"
            r"|what\s+are\s+the\s+risks?|relative\s+to|consider\s+when)\b",
        ],
        "message": (
            "I'm not able to recommend concentrating all savings in highly volatile assets, "
            "using leverage for single-stock positions, or risking emergency funds in options. "
            "These strategies carry a high probability of catastrophic losses."
        ),
    },
    "sanctions_evasion": {
        "block_patterns": [
            r"\b(bypass\s+ofac|evade\s+sanctions?"
            r"|route\s+(trades?|funds?)\s+through\s+a\s+shell"
            r"|sanctioned\s+(russian|iranian|north\s+korean|entity|company)"
            r"\s+without\s+it\s+being\s+traced"
            r"|invest\s+in\s+a\s+sanctioned"
            r"|circumvent\s+sanctions?)\b",
        ],
        "pass_patterns": [
            r"\b(what\s+(are|is)|explain|ofac\s+sanctions?\s+and"
            r"|compliance\s+obligations?|brokers?\s+(have|screen)"
            r"|how\s+do\s+brokerages?|definition|regulations?)\b",
        ],
        "message": (
            "I can't assist with circumventing sanctions or investing in sanctioned entities. "
            "Sanctions violations carry severe civil and criminal penalties under OFAC, EU, "
            "and UN frameworks. Please consult a compliance specialist."
        ),
    },
    "fraud": {
        "block_patterns": [
            r"\b(fake\s+(contract|invoice|statement|document|note)"
            r"|forge\s+|fabricate\s+|falsify\s+)\b",
        ],
        "pass_patterns": [],
        "message": (
            "I'm unable to assist with creating fraudulent documents or records. "
            "This constitutes financial fraud and is a criminal offence."
        ),
    },
}


def _compile(patterns: list[str]) -> re.Pattern | None:
    if not patterns:
        return None
    combined = "|".join(f"(?:{p})" for p in patterns)
    return re.compile(combined, re.IGNORECASE | re.DOTALL)


_COMPILED: dict[str, dict] = {
    cat: {
        "block": _compile(cfg["block_patterns"]),
        "pass":  _compile(cfg["pass_patterns"]),
        "message": cfg["message"],
    }
    for cat, cfg in _CATEGORIES.items()
}


def check(query: str) -> SafetyVerdict:
    """
    Run the safety guard.
    Pure local computation — no I/O, no LLM call.
    """
    q = query.strip()
    for category, compiled in _COMPILED.items():
        block_match = compiled["block"] and compiled["block"].search(q)
        if block_match:
            pass_match = compiled["pass"] and compiled["pass"].search(q)
            if pass_match:
                continue
            return SafetyVerdict(
                blocked=True,
                category=category,
                message=compiled["message"],
            )
    return SafetyVerdict(blocked=False)
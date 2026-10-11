"""Source reliability classifier: official / authoritative / media / community / forum / unknown."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

OFFICIAL_DOMAINS = {
    "gov", "edu", "mil", "gov.uk", "go.jp", "gob.mx", "europa.eu", "who.int",
    "un.org", "unicef.org", "unhcr.org", "fao.org", "imf.org", "worldbank.org",
    "wto.org", "oecd.org", "esa.int", "nasa.gov", "nih.gov", "cdc.gov",
    "fda.gov", "nsa.gov", "parl.uk", "congress.gov", "judiciary.house.gov",
}
AUTHORITATIVE_DOMAINS = {
    "wikipedia.org", "arxiv.org", "doi.org", "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov", "scholar.google.com", "plos.org", "nature.com",
    "sciencedirect.com", "springer.com", "acs.org", "ieee.org",
    "frontiersin.org", "protocols.io", "semanticscholar.org",
    "docs.python.org", "developer.mozilla.org", "react.dev", "kubernetes.io",
    "openai.com", "anthropic.com", "deepmind.google", "huggingface.co",
    "gitlab.com", "npmjs.com", "pypi.org",
}
MEDIA_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "nytimes.com",
    "washingtonpost.com", "theguardian.com", "bloomberg.com", "ft.com",
    "economist.com", "wsj.com", "cnbc.com", "npr.org", "theverge.com",
    "arstechnica.com", "wired.com", "techcrunch.com", "engadget.com",
    "zdnet.com", "cnet.com", "mashable.com",
}
COMMUNITY_DOMAINS = {
    "stackoverflow.com", "github.com", "stackexchange.com", "serverfault.com",
    "superuser.com", "quora.com", "researchgate.net",
}
FORUM_DOMAINS = {
    "reddit.com", "twitter.com", "x.com", "ycombinator.com", "news.ycombinator.com",
    "facebook.com", "instagram.com", "tiktok.com", "vk.com", "discord.com",
    "t.me", "telegram.org", "forum.xda-developers.com", "medium.com",
    "truthsocial.com", "gab.com", "4chan.org", "8kun.top",
}

COMPLAINT_MARKERS = (
    "scam", "fraud", "complaint", "scamalert", "refund", "chargeback",
    "didn't work", "does not work", "broke", "total garbage", "warning",
    "avoid", "lawsuit", "fake",
)


@dataclass(slots=True)
class Verdict:
    category: str
    tier: str
    trust: float
    label: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "tier": self.tier,
            "trust": self.trust,
            "label": self.label,
        }


def _domain(url: str) -> str:
    host = re.sub(r"^[a-z]+://", "", url.lower())
    host = host.split("/")[0].split("?")[0].split("@")[-1]
    host = host.split(":")[0]
    return host.strip()


def classify(url: str, source: str = "", title: str = "", text: str = "") -> Verdict:
    domain = _domain(url or "")
    if not domain:
        return Verdict("unknown", "unknown", 0.1, "no domain")
    labels = domain.split(".")
    base = ".".join(labels[-2:]) if len(labels) >= 2 else domain
    tld = labels[-1]

    if base in OFFICIAL_DOMAINS or tld in ("gov", "edu", "mil"):
        return Verdict("official", "official", 0.95, "official / governmental")
    if base in AUTHORITATIVE_DOMAINS:
        return Verdict("authoritative", "authoritative", 0.93, "authoritative source")
    if base in MEDIA_DOMAINS:
        return Verdict("media", "media", 0.7, "established media")
    if base in COMMUNITY_DOMAINS:
        return Verdict("community", "community", 0.5, "community / Q&A / code")
    if base in FORUM_DOMAINS:
        return Verdict("forum_social", "forum_social", 0.25, "forum / social")

    haystack = f"{source} {title} {text}".lower()
    if any(marker in haystack for marker in COMPLAINT_MARKERS):
        return Verdict("complaint_flagged", "forum_social", 0.15, "complaint-flagged")

    if tld in ("com", "org", "net", "io", "dev", "ai", "app") and len(labels) >= 2:
        return Verdict("unknown_owned", "unknown", 0.35, "unknown / minor site")
    return Verdict("unknown", "unknown", 0.1, "unknown source")


def rank(docs: list[Any], key_url: str = "url", key_source: str = "source",
         key_title: str = "title", key_text: str = "text") -> list[dict[str, Any]]:
    ranked = []
    for doc in docs:
        url = getattr(doc, key_url, "") if hasattr(doc, key_url) else doc.get(key_url, "")
        source = getattr(doc, key_source, "") if hasattr(doc, key_source) else doc.get(key_source, "")
        title = getattr(doc, key_title, "") if hasattr(doc, key_title) else doc.get(key_title, "")
        text = getattr(doc, key_text, "") if hasattr(doc, key_text) else doc.get(key_text, "")
        verdict = classify(url, source, title, text)
        item = doc.to_dict() if hasattr(doc, "to_dict") else doc
        ranked.append({"doc": doc, "verdict": verdict, "url": url})
    ranked.sort(key=lambda r: r["verdict"].trust, reverse=True)
    return ranked
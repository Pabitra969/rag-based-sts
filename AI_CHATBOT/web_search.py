import asyncio
import json
import os
from dataclasses import dataclass
from typing import List
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


DEFAULT_SEARCH_URL = "https://api.duckduckgo.com/"


@dataclass
class WebResult:
    title: str
    snippet: str
    source: str
    url: str = ""


def web_search_enabled() -> bool:
    raw = os.environ.get("WEB_SEARCH_ENABLED", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _source_name(url: str, fallback: str = "Web") -> str:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        host = ""
    if not host:
        return fallback
    host = host.removeprefix("www.")
    return host or fallback


def _flatten_related_topics(items):
    flat = []
    for item in items or []:
        if isinstance(item, dict) and "Topics" in item:
            flat.extend(_flatten_related_topics(item.get("Topics")))
        else:
            flat.append(item)
    return flat


def _search_sync(query: str, max_results: int = 3) -> List[WebResult]:
    params = urlencode(
        {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
            "no_redirect": "1",
        }
    )
    req = Request(
        f"{os.environ.get('WEB_SEARCH_API_URL', DEFAULT_SEARCH_URL)}?{params}",
        headers={"User-Agent": "AriaLocalBot/1.0"},
    )

    with urlopen(req, timeout=float(os.environ.get("WEB_SEARCH_TIMEOUT_SEC", "8"))) as res:
        payload = json.loads(res.read().decode("utf-8", errors="replace"))

    results: List[WebResult] = []

    answer = str(payload.get("Answer", "")).strip()
    answer_type = str(payload.get("AnswerType", "")).strip()
    abstract = str(payload.get("AbstractText", "")).strip()
    abstract_url = str(payload.get("AbstractURL", "")).strip()
    heading = str(payload.get("Heading", "")).strip() or query.strip()
    abstract_source = str(payload.get("AbstractSource", "")).strip() or _source_name(abstract_url, "DuckDuckGo")

    if answer and answer_type:
        results.append(
            WebResult(
                title=heading or "Direct answer",
                snippet=answer,
                source=abstract_source or "DuckDuckGo",
                url=abstract_url,
            )
        )

    if abstract:
        results.append(
            WebResult(
                title=heading or "Summary",
                snippet=abstract,
                source=abstract_source or "DuckDuckGo",
                url=abstract_url,
            )
        )

    for item in _flatten_related_topics(payload.get("RelatedTopics", [])):
        if len(results) >= max_results:
            break
        if not isinstance(item, dict):
            continue
        text = str(item.get("Text", "")).strip()
        url = str(item.get("FirstURL", "")).strip()
        if not text:
            continue
        title = text.split(" - ", 1)[0].strip()
        snippet = text.split(" - ", 1)[-1].strip() if " - " in text else text
        results.append(
            WebResult(
                title=title[:80],
                snippet=snippet[:220],
                source=_source_name(url, "DuckDuckGo"),
                url=url,
            )
        )

    deduped: List[WebResult] = []
    seen = set()
    for item in results:
        key = (item.title.lower(), item.snippet.lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= max_results:
            break

    return deduped


async def search_web(query: str, max_results: int = 3) -> List[WebResult]:
    if not web_search_enabled():
        return []
    return await asyncio.to_thread(_search_sync, query, max_results)


def compress_web_results(results: List[WebResult], max_chars: int = 560) -> str:
    lines = []
    total = 0
    for idx, item in enumerate(results, start=1):
        line = f"{idx}. {item.source}: {item.title}. {item.snippet}".strip()
        line = line.replace("\n", " ")
        if len(line) > 220:
            line = f"{line[:217].rstrip()}..."
        extra = len(line) + (1 if lines else 0)
        if total + extra > max_chars:
            break
        lines.append(line)
        total += extra
    return "\n".join(lines)

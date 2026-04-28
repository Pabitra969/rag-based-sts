import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import List
from urllib.parse import parse_qs, quote, unquote, urlencode, urlparse
from urllib.request import Request, urlopen


DEFAULT_SEARCH_URL = "https://api.duckduckgo.com/"
DEFAULT_SEARCH_HTML_URL = "https://html.duckduckgo.com/html/"
DEFAULT_WIKIPEDIA_SEARCH_URL = "https://en.wikipedia.org/w/api.php"


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


def _html_to_text(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value or "")
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    return re.sub(r"\s+", " ", text).strip()


def _clean_result_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if url.startswith("//"):
        url = f"https:{url}"
    try:
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        if parsed.netloc.endswith("duckduckgo.com") and "uddg" in query:
            return unquote(query["uddg"][0])
    except Exception:
        return url
    return url


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


def _search_wikipedia_sync(query: str, max_results: int = 3) -> List[WebResult]:
    params = urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "utf8": "1",
            "format": "json",
            "srlimit": str(max_results),
        }
    )
    req = Request(
        f"{os.environ.get('WEB_SEARCH_WIKIPEDIA_URL', DEFAULT_WIKIPEDIA_SEARCH_URL)}?{params}",
        headers={"User-Agent": "AriaLocalBot/1.0"},
    )

    with urlopen(req, timeout=float(os.environ.get("WEB_SEARCH_TIMEOUT_SEC", "8"))) as res:
        payload = json.loads(res.read().decode("utf-8", errors="replace"))

    items = payload.get("query", {}).get("search", []) or []
    results: List[WebResult] = []

    for item in items:
        title = str(item.get("title", "")).strip()
        snippet = _html_to_text(str(item.get("snippet", "")).strip())
        if not title:
            continue
        page_url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
        results.append(
            WebResult(
                title=title[:120],
                snippet=snippet[:220],
                source="Wikipedia",
                url=page_url,
            )
        )
        if len(results) >= max_results:
            break

    return results


def _search_html_fallback_sync(query: str, max_results: int = 3) -> List[WebResult]:
    params = urlencode({"q": query, "kl": "us-en"})
    req = Request(
        f"{os.environ.get('WEB_SEARCH_HTML_URL', DEFAULT_SEARCH_HTML_URL)}?{params}",
        headers={"User-Agent": "AriaLocalBot/1.0"},
    )

    with urlopen(req, timeout=float(os.environ.get("WEB_SEARCH_TIMEOUT_SEC", "8"))) as res:
        html = res.read().decode("utf-8", errors="replace")

    title_matches = list(
        re.finditer(
            r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>',
            html,
            flags=re.I | re.S,
        )
    )
    snippet_matches = list(
        re.finditer(
            r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(?P<snippet>.*?)</a>|'
            r'<td[^>]+class="[^"]*result-snippet[^"]*"[^>]*>(?P<snippet_td>.*?)</td>',
            html,
            flags=re.I | re.S,
        )
    )

    results: List[WebResult] = []
    for idx, match in enumerate(title_matches):
        if len(results) >= max_results:
            break
        raw_url = match.group("url") or ""
        raw_title = match.group("title") or ""
        snippet_match = snippet_matches[idx] if idx < len(snippet_matches) else None
        raw_snippet = ""
        if snippet_match:
            raw_snippet = snippet_match.group("snippet") or snippet_match.group("snippet_td") or ""

        clean_url = _clean_result_url(raw_url)
        title = _html_to_text(raw_title)[:120]
        snippet = _html_to_text(raw_snippet)[:220]

        if not title:
            continue

        results.append(
            WebResult(
                title=title,
                snippet=snippet,
                source=_source_name(clean_url, "DuckDuckGo"),
                url=clean_url,
            )
        )

    return results


async def search_web(query: str, max_results: int = 3) -> List[WebResult]:
    if not web_search_enabled():
        return []
    merged: List[WebResult] = []
    seen = set()

    for label, loader in [
        ("wikipedia lookup", _search_wikipedia_sync),
        ("instant answer lookup", _search_sync),
        ("html fallback lookup", _search_html_fallback_sync),
    ]:
        if len(merged) >= max_results:
            break
        try:
            batch = await asyncio.to_thread(loader, query, max_results)
        except Exception as exc:
            print(f"[WEB SEARCH] {label} failed: {exc}")
            continue

        for item in batch:
            key = ((item.url or "").lower(), item.title.lower(), item.snippet.lower())
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
            if len(merged) >= max_results:
                break

    return merged


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

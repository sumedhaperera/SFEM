# phase1_ingestion/crawl_sf_flow.py
from __future__ import annotations

import argparse, time, json, hashlib, os
from urllib.parse import urljoin, urlparse, urlunparse, parse_qs, urlencode
from datetime import datetime, timezone
from typing import Set, List, Dict, Tuple, Optional
import requests
from bs4 import BeautifulSoup, Tag
from urllib import robotparser
from tqdm import tqdm
import re
import email.utils as eut

# --- robust user-agent ---
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# ---------------- Chunker (inline, no external import) ----------------
HEADING_TAGS = ["h1", "h2", "h3"]

def _to_ts(iso_or_httpdate: str | None) -> float | None:
    if not iso_or_httpdate:
        return None
    # try ISO
    try:
        return datetime.fromisoformat(iso_or_httpdate.replace("Z","+00:00")).timestamp()
    except Exception:
        pass
    # try HTTP-date (rare)
    try:
        return eut.parsedate_to_datetime(iso_or_httpdate).timestamp()
    except Exception:
        return None


def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _extract_release_train(soup: BeautifulSoup) -> Optional[str]:
    text = soup.get_text(" ", strip=True)
    m = re.search(r"(Spring|Summer|Winter)\s+'?(\d{2})", text, re.I)
    if m:
        return f"{m.group(1).title()} '{m.group(2)}"
    return None

def _choose_main(soup: BeautifulSoup) -> Tag:
    for sel in ["main", "article", "#main-content", "#content", ".slds-col--padded"]:
        el = soup.select_one(sel)
        if el:
            return el
    return soup.body or soup

def chunk_article(html: str) -> Tuple[List[Dict], Dict]:
    soup = BeautifulSoup(html, "lxml")
    main = _choose_main(soup)
    release_train = _extract_release_train(soup)
    anchors: List[str] = []
    chunks: List[Dict] = []

    current = {"level": None, "headings_path": [], "parts": []}

    def flush():
        if not current["parts"] and not current["headings_path"]:
            return
        text = _clean_text(
            " ".join(
                p.get_text(" ", strip=True) if hasattr(p, "get_text") else str(p)
                for p in current["parts"]
            )
        )
        anchor = None
        for el in reversed(current["parts"]):
            if isinstance(el, Tag) and el.name in HEADING_TAGS and el.get("id"):
                anchor = el["id"]
                break
        chunks.append(
            {"headings_path": list(current["headings_path"]), "anchor": anchor, "text": text}
        )
        current["level"] = None
        current["headings_path"] = []
        current["parts"] = []

    def start_section(h: Tag):
        if current["parts"]:
            flush()
        title = _clean_text(h.get_text(" ", strip=True))
        level = int(h.name[1])  # 1/2/3
        if h.get("id"):
            anchors.append(h["id"])
        if level == 1:
            path = [title]
        elif level == 2:
            path = [current["headings_path"][0], title] if current["headings_path"] else [title]
        else:  # h3
            base = current["headings_path"][:2] if current["headings_path"] else []
            if not base:
                base = ["(untitled)"]
            path = base + [title]
        current["level"] = level
        current["headings_path"] = path
        current["parts"] = [h]

    # Walk DOM
    for el in main.descendants:
        if not isinstance(el, Tag):
            continue
        name = el.name.lower()
        if name in HEADING_TAGS:
            start_section(el)
        elif name in ["p", "ul", "ol", "pre", "code", "table", "blockquote", "dl"]:
            if current["headings_path"]:
                current["parts"].append(el)

    if current["parts"]:
        flush()

    # Fallbacks: ensure at least 1 chunk if there is any text
    if not chunks:
        blocks = [
            el.get_text(" ", strip=True)
            for el in main.find_all(["p", "ul", "ol", "pre", "code", "table", "blockquote", "dl"])
        ]
        text = _clean_text(" ".join(blocks))
        if text:
            chunks.append({"headings_path": ["(untitled)"], "anchor": None, "text": text})
    if not chunks:
        text = _clean_text(main.get_text(" ", strip=True))
        if text:
            chunks.append({"headings_path": ["(untitled)"], "anchor": None, "text": text})

    meta = {"release_train": release_train, "anchors": anchors}
    return chunks, meta

# ---------------- Crawler ----------------
def canonicalize(url: str) -> str:
    u = urlparse(url)
    qs = parse_qs(u.query)
    keep = {k: v for k, v in qs.items() if k.lower() in {"id", "type", "language", "bundleid", "r"}}
    return urlunparse((u.scheme, u.netloc, u.path, "", urlencode(keep, doseq=True), ""))

def robots(anchor_url: str) -> robotparser.RobotFileParser:
    p = urlparse(anchor_url)
    robots_url = f"{p.scheme}://{p.netloc}/robots.txt"
    rp = robotparser.RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
    except Exception:
        pass
    return rp

def discover(html: str, base_url: str, path_prefix: str):
    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        if path_prefix in urlparse(href).path:
            yield canonicalize(href)

def fetch(url: str):
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r

def build_id(url: str, anchor: str | None) -> str:
    h = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
    return f"sf_help:flow:{h}#{anchor}" if anchor else f"sf_help:flow:{h}"

def crawl(args):
    prefix = args.path_prefix
    out = open(args.out, "w", encoding="utf-8")

    robots_anchor = (args.seeds[0] if args.seeds else args.base) or "https://help.salesforce.com/s/articleView"
    rp = robots(robots_anchor)

    queue: List[str] = []
    seen: Set[str] = set()

    # Seeds
    if args.seeds:
        for s in args.seeds:
            if prefix in urlparse(s).path:
                u = canonicalize(s)
                if u not in seen:
                    seen.add(u)
                    queue.append(u)

    # Optional discovery from base
    if args.base:
        try:
            r = fetch(args.base)
            for link in discover(r.text, args.base, prefix):
                if link not in seen:
                    seen.add(link)
                    queue.append(link)
        except Exception as e:
            print(f"[seed] failed base fetch {args.base}: {e}")

    if not queue:
        print("[exit] No initial URLs to crawl. Provide --seeds or a base page that links to your path-prefix.")
        out.close()
        return

    debug_dir = None
    if args.debug_dump > 0:
        debug_dir = os.path.join(os.path.dirname(args.out) or ".", "debug_html")
        os.makedirs(debug_dir, exist_ok=True)

    pbar, count = tqdm(total=args.max_pages, desc="Crawling Flow docs"), 0
    while queue and count < args.max_pages:
        url = queue.pop(0)
        try:
            if not rp.can_fetch(HEADERS["User-Agent"], url):
                continue
        except Exception:
            pass

        try:
            resp = fetch(url)
            html = resp.text
        except Exception as e:
            print(f"[skip] {url}: {e}")
            continue

        # Debug dump (first N pages)
        if debug_dir and count < args.debug_dump:
            fname = os.path.join(debug_dir, f"page_{count+1}.html")
            with open(fname, "w", encoding="utf-8") as f:
                f.write(html)

        soup = BeautifulSoup(html, "lxml")
        title = soup.title.get_text(strip=True) if soup.title else None
        lastmod = None
        for m in soup.find_all("meta"):
            if m.get("itemprop") == "dateModified" and m.get("content"):
                lastmod = m["content"]
                break

        chunks, meta = chunk_article(html)
        ts = datetime.now(timezone.utc).isoformat()

        # Always emit at least one chunk if any text exists
        if not chunks:
            txt = _clean_text(soup.get_text(" ", strip=True))
            if txt:
                chunks = [{"headings_path": ["(untitled)"], "anchor": None, "text": txt}]

        doc_ts = _to_ts(lastmod) or datetime.now(timezone.utc).timestamp()
        # Write chunks
        wrote_any = False
        for ch in chunks:
            out.write(
                json.dumps(
                    {
                        "id": build_id(url, ch.get("anchor")),
                        "doc_id": (urlparse(url).path.split("/")[-1] or "flow_doc").split(".")[0],
                        "url": url,
                        "title": title,
                        "headings_path": ch["headings_path"],
                        "anchor": ch.get("anchor"),
                        "text": ch["text"],
                        "source": "sf_help",
                        "product": "flow",
                        "release_train": meta.get("release_train"),
                        "lastmod": lastmod,
                        "doc_ts": doc_ts, 
                        "anchors": meta.get("anchors") or [],
                        "crawl_ts": ts,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            wrote_any = True

        # Proceed even if wrote_any is False (might be a bad page)
        count += 1
        pbar.update(1)

        # Discover more links from this page
        for link in discover(html, url, prefix):
            if link not in seen:
                seen.add(link)
                queue.append(link)

        time.sleep(args.delay)

    out.close()
    pbar.close()
    print(f"[done] wrote → {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", help="Optional base page to discover links from")
    ap.add_argument("--seeds", nargs="*", help="One or more starting article URLs under the path-prefix")
    ap.add_argument("--path-prefix", required=True, help="Restrict crawl to paths containing this substring")
    ap.add_argument("--max-pages", type=int, default=1000)
    ap.add_argument("--delay", type=float, default=0.5)
    ap.add_argument("--out", default="data/sf_flow_chunks.jsonl")
    ap.add_argument("--debug-dump", type=int, default=2, help="Dump raw HTML for first N pages into data/debug_html/")
    crawl(ap.parse_args())

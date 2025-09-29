# phase1_ingestion/release_notes_ingest.py
from __future__ import annotations
import argparse, json, os, hashlib
from datetime import datetime, timezone
from typing import List, Dict
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from html_chunker import chunk_article  # uses your updated chunker

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

def fetch(u: str) -> str:
    r = requests.get(u, headers=HEADERS, timeout=25)
    r.raise_for_status()
    return r.text

def build_id(url: str, anchor: str|None) -> str:
    h = hashlib.md5(url.encode("utf-8")).hexdigest()[:12]
    return f"sf_release_notes:flow:{h}#{anchor or 'root'}"

def ingest(args):
    seeds: List[str] = args.seeds or []
    if not seeds:
        print("[exit] no seeds provided (use --seeds <urls...>)")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_f = open(args.out, "w", encoding="utf-8")
    crawl_ts = datetime.now(timezone.utc).isoformat()

    # optional debug
    dbg_dir = None
    if args.debug_dump > 0:
        dbg_dir = os.path.join(os.path.dirname(args.out) or ".", "debug_html")
        os.makedirs(dbg_dir, exist_ok=True)

    total = 0
    for i, url in enumerate(tqdm(seeds, desc="Release note pages")):
        try:
            html = fetch(url)
        except Exception as e:
            print(f"[skip] {url}: {e}")
            continue

        if dbg_dir and i < args.debug_dump:
            with open(os.path.join(dbg_dir, f"rn_{i+1}.html"), "w", encoding="utf-8") as f:
                f.write(html)

        soup = BeautifulSoup(html, "lxml")
        title = soup.title.get_text(strip=True) if soup.title else None

        # attempt to pull lastmod from meta if present
        lastmod = None
        for m in soup.find_all("meta"):
            if m.get("itemprop") == "dateModified" and m.get("content"):
                lastmod = m["content"]; break

        chunks, meta = chunk_article(html)

        wrote = 0
        for ch in chunks:
            out_f.write(json.dumps({
                "id": build_id(url, ch.get("anchor")),
                "doc_id": (urlparse(url).path.split("/")[-1] or "release_notes").split(".")[0],
                "url": url,
                "title": title,
                "headings_path": ch["headings_path"],
                "anchor": ch.get("anchor"),
                "text": ch["text"],
                "source": "sf_release_notes",
                "product": "flow",
                "release_train": meta.get("release_train"),
                "lastmod": lastmod,
                "crawl_ts": crawl_ts,
            }, ensure_ascii=False) + "\n")
            wrote += 1

        total += wrote
        if wrote == 0:
            # as last resort, emit one chunk with page text
            txt = " ".join(soup.stripped_strings)
            if txt:
                out_f.write(json.dumps({
                    "id": build_id(url, None),
                    "doc_id": (urlparse(url).path.split("/")[-1] or "release_notes").split(".")[0],
                    "url": url,
                    "title": title,
                    "headings_path": ["(untitled)"],
                    "anchor": None,
                    "text": txt[:20000],
                    "source": "sf_release_notes",
                    "product": "flow",
                    "release_train": meta.get("release_train"),
                    "lastmod": lastmod,
                    "crawl_ts": crawl_ts,
                }, ensure_ascii=False) + "\n")
                total += 1

    out_f.close()
    print(f"[done] wrote → {args.out} ({total} chunks)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Flow Release Notes → JSONL")
    ap.add_argument("--seeds", nargs="*", help="Explicit release note URLs to ingest")
    ap.add_argument("--out", default="data/flow_releasenotes.jsonl")
    ap.add_argument("--debug-dump", type=int, default=1)
    ingest(ap.parse_args())

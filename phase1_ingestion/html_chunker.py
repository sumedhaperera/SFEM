from __future__ import annotations

from bs4 import BeautifulSoup, Tag
from typing import List, Dict, Tuple, Optional
import re

# Accept top-level page headings too
HEADING_TAGS = ["h1", "h2", "h3"]

def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _extract_release_train(soup: BeautifulSoup) -> Optional[str]:
    """
    Try to find "Spring '25", "Summer '25", "Winter '26" in the page text.
    """
    text = soup.get_text(" ", strip=True)
    m = re.search(r"(Spring|Summer|Winter)\s+'?(\d{2})", text, re.I)
    if m:
        return f"{m.group(1).title()} '{m.group(2)}"
    return None

def _choose_main(soup: BeautifulSoup) -> Tag:
    # Try common containers; fall back to body
    for sel in ["main", "article", "#main-content", "#content", ".slds-col--padded"]:
        el = soup.select_one(sel)
        if el:
            return el
    return soup.body or soup

def chunk_article(html: str) -> Tuple[List[Dict], Dict]:
    """
    Split article into chunks by h1/h2/h3 while preserving nearby content.
    Returns (chunks, meta_from_page)
    """
    soup = BeautifulSoup(html, "lxml")
    main = _choose_main(soup)

    release_train = _extract_release_train(soup)
    anchors: List[str] = []
    chunks: List[Dict] = []

    current = {
        "level": None,                # 1, 2, 3 for h1/h2/h3
        "headings_path": [],          # ["H1","H2"] etc.
        "parts": []                   # collected Tag parts for this section
    }

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
            {
                "headings_path": list(current["headings_path"]),
                "anchor": anchor,
                "text": text,
            }
        )
        current["level"] = None
        current["headings_path"] = []
        current["parts"] = []

    def start_section(h: Tag):
        # Close previous
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
        else:  # level 3
            base = current["headings_path"][:2] if current["headings_path"] else []
            if not base:
                base = ["(untitled)"]
            path = base + [title]

        current["level"] = level
        current["headings_path"] = path
        current["parts"] = [h]

    # Walk content in document order
    for el in main.descendants:
        if not isinstance(el, Tag):
            continue
        name = el.name.lower()
        if name in HEADING_TAGS:
            start_section(el)
        elif name in ["p", "ul", "ol", "pre", "code", "table", "blockquote", "dl"]:
            if current["headings_path"]:
                current["parts"].append(el)

    # Flush last open section
    if current["parts"]:
        flush()

    # Fallback 1: no heading sections → collect block-level content
    if not chunks:
        blocks = []
        for el in main.find_all(["p", "ul", "ol", "pre", "code", "table", "blockquote", "dl"]):
            blocks.append(el.get_text(" ", strip=True))
        text = _clean_text(" ".join(blocks))
        if text:
            chunks.append({"headings_path": ["(untitled)"], "anchor": None, "text": text})

    # Fallback 2: still empty → use all main text
    if not chunks:
        text = _clean_text(main.get_text(" ", strip=True))
        if text:
            chunks.append({"headings_path": ["(untitled)"], "anchor": None, "text": text})

    meta = {"release_train": release_train, "anchors": anchors}
    return chunks, meta

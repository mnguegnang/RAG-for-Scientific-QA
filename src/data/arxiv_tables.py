"""
Table bodies for QASPER papers, extracted from their arXiv LaTeX sources.

Why: 40% of the evaluation questions have table/figure ("FLOAT SELECTED")
gold evidence, and 72% of those reference answers are numbers that live in a
table body, not in the caption (key_metrics_improvements.md, 2026-10-03, P2).
QASPER only ships captions and page images, but every QASPER paper was
selected *because* it has an arXiv LaTeX source (Dasigi et al. 2021, §2.1), so
the table text can be recovered from `tabular` environments.

Output: a JSON file {paper_id: {qasper_caption: markdown_table}} consumed by
src/pipeline_ingest.py, so ingestion itself needs no network access.

Usage:
    python -m src.data.arxiv_tables                       # all train papers
    python -m src.data.arxiv_tables --limit 20            # quick trial
    python -m src.data.arxiv_tables --out data/table_bodies.json --delay 3

Network etiquette: arXiv asks automated clients for at most one request every
three seconds; --delay defaults to 3.0 and only applies to cache misses.
Sources are cached under data/arxiv_src/ as gzipped LaTeX (figures dropped).
"""
import argparse
import gzip
import io
import json
import logging
import re
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARXIV_SRC_URL = "https://export.arxiv.org/src/{arxiv_id}"
USER_AGENT = "RAG-for-Scientific-QA/table-extraction (research use; arXiv src endpoint)"

# ── Fetching ──────────────────────────────────────────────────────────────────

def extract_tex(raw: bytes) -> str:
    """Concatenate every .tex file in an arXiv source payload ('' if none)."""
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tar:
            parts = []
            for member in tar.getmembers():
                if member.isfile() and member.name.lower().endswith(".tex"):
                    fh = tar.extractfile(member)
                    if fh is not None:
                        parts.append(fh.read().decode("utf-8", errors="replace"))
            return "\n".join(parts)
    except tarfile.TarError:
        pass
    # Single-file submissions are a gzipped .tex, not a tarball.
    try:
        text = gzip.decompress(raw).decode("utf-8", errors="replace")
    except (OSError, EOFError):
        return ""
    return text if ("\\begin" in text or "\\documentclass" in text) else ""


def fetch_tex(arxiv_id: str, cache_dir: Path, delay: float = 3.0,
              max_retries: int = 3) -> str:
    """LaTeX source for *arxiv_id*, from cache or arXiv. '' when unavailable."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{arxiv_id}.tex.gz"
    missing = cache_dir / f"{arxiv_id}.nosrc"
    if cached.exists():
        return gzip.decompress(cached.read_bytes()).decode("utf-8", errors="replace")
    if missing.exists():
        return ""

    url = ARXIV_SRC_URL.format(arxiv_id=arxiv_id)
    wait = delay
    for attempt in range(1, max_retries + 1):
        time.sleep(delay)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                missing.touch()
                return ""
            logger.warning("%s: HTTP %s (attempt %d/%d)", arxiv_id, exc.code, attempt, max_retries)
        except (urllib.error.URLError, TimeoutError) as exc:
            logger.warning("%s: %s (attempt %d/%d)", arxiv_id, exc, attempt, max_retries)
        time.sleep(wait)
        wait *= 2
    else:
        return ""  # transient failure: not cached, retried on the next run

    tex = extract_tex(raw)
    if tex:
        cached.write_bytes(gzip.compress(tex.encode("utf-8")))
    else:
        missing.touch()  # PDF-only submission
    return tex


# ── LaTeX parsing ─────────────────────────────────────────────────────────────

def strip_comments(tex: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", tex)


def read_group(s: str, i: int, open_ch: str = "{", close_ch: str = "}") -> Tuple[str, int]:
    """
    s[i] must be *open_ch*; returns (content, index after the closing char).
    Escaped braces (\\{ \\}) do not count.
    """
    assert s[i] == open_ch
    depth, j = 0, i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return s[i + 1:], len(s)


def _skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i].isspace():
        i += 1
    return i


def _skip_args(s: str, i: int, n_required: int) -> int:
    """Skip optional [..] groups and *n_required* {..} groups starting at i."""
    seen = 0
    while True:
        i = _skip_ws(s, i)
        if i < len(s) and s[i] == "[":
            _, i = read_group(s, i, "[", "]")
        elif seen < n_required and i < len(s) and s[i] == "{":
            _, i = read_group(s, i)
            seen += 1
        else:
            return i


# Commands whose (last) argument is kept as text.
_KEEP_ARG = ("textbf", "textit", "emph", "underline", "mathbf", "mathrm", "mathit",
             "text", "textsc", "texttt", "textrm", "textsf", "mbox", "boldsymbol",
             "bm", "textnormal", "uline", "makecell", "shortstack", "hbox",
             "textup", "mathsf", "operatorname", "sc", "bf", "it")
# Commands dropped together with their arguments.
_DROP_WITH_ARGS = {"cite": 1, "citep": 1, "citet": 1, "ref": 1, "label": 1,
                   "cellcolor": 1, "rowcolor": 1, "color": 1, "textcolor": 1,
                   "vspace": 1, "hspace": 1, "cline": 1, "cmidrule": 1, "hhline": 1,
                   "specialrule": 3, "addlinespace": 0, "footnote": 1, "tnote": 1,
                   "includegraphics": 1, "rule": 2, "arrayrulecolor": 1}
_SYMBOLS = {"pm": "±", "times": "×", "uparrow": "↑", "downarrow": "↓",
            "dagger": "†", "ddagger": "‡", "ast": "*", "star": "*", "approx": "≈",
            "leq": "≤", "geq": "≥", "le": "≤", "ge": "≥", "rightarrow": "→",
            "checkmark": "✓", "cdot": "·", "sim": "~", "ldots": "...", "dots": "..."}


def latex_to_text(s: str) -> str:
    """Best-effort plain text for a caption or table cell."""
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            m = re.match(r"\\([a-zA-Z]+)\*?", s[i:])
            if not m:  # escaped symbol: \% \& \_ \# \$ \{ \}
                if i + 1 < len(s):
                    out.append(s[i + 1] if s[i + 1] in "%&_#${}" else " ")
                i += 2
                continue
            name, i = m.group(1), i + m.end()
            if name in _DROP_WITH_ARGS:
                i = _skip_args(s, i, _DROP_WITH_ARGS[name])
            elif name in ("multicolumn", "multirow"):
                i = _skip_args(s, i, 2)            # keep the 3rd group as text
            elif name in _SYMBOLS:
                out.append(_SYMBOLS[name])
            elif name not in _KEEP_ARG:
                # Unknown macro or a switch (\small, \centering, \hline, ...):
                # drop the name; any {argument} that follows is kept as text,
                # which is right far more often than wrong in table cells.
                out.append(" ")
            continue
        if c in "{}$":
            i += 1
            continue
        if c == "~":
            out.append(" ")
            i += 1
            continue
        out.append(c)
        i += 1
    return re.sub(r"\s+", " ", "".join(out)).strip()


_TABLE_ENV_RE = re.compile(r"\\begin\{((?:sideways)?table\*?)\}")
_TABULAR_RE = re.compile(r"\\begin\{(tabular\*?|tabularx|tabulary|longtable\*?)\}")
# Required argument count before the body: tabular{spec}; tabular*/x/y{width}{spec}.
_TABULAR_ARGS = {"tabular": 1, "longtable": 1, "longtable*": 1,
                 "tabular*": 2, "tabularx": 2, "tabulary": 2}
_ROW_SEP_RE = re.compile(r"\\\\(?:\s*\[[^\]]*\])?|\\tabularnewline")
_CELL_SEP_RE = re.compile(r"(?<!\\)&")


def parse_tabular_rows(body: str) -> List[List[str]]:
    rows = []
    for raw_row in _ROW_SEP_RE.split(body):
        cells = [latex_to_text(cell) for cell in _CELL_SEP_RE.split(raw_row)]
        if any(cells):
            rows.append(cells)
    return rows


def extract_tables(tex: str) -> List[Dict]:
    """[{'caption': plain-text caption, 'rows': [[cell, ...], ...]}, ...]"""
    tex = strip_comments(tex)
    tables = []
    for env in _TABLE_ENV_RE.finditer(tex):
        name = env.group(1)
        end = tex.find(f"\\end{{{name}}}", env.end())
        block = tex[env.end(): end if end != -1 else len(tex)]

        caption = ""
        cap = re.search(r"\\caption\*?", block)
        if cap:
            j = _skip_ws(block, cap.end())
            if j < len(block) and block[j] == "[":
                _, j = read_group(block, j, "[", "]")
                j = _skip_ws(block, j)
            if j < len(block) and block[j] == "{":
                caption = latex_to_text(read_group(block, j)[0])

        rows = []
        for tab in _TABULAR_RE.finditer(block):
            kind = tab.group(1)
            start = _skip_args(block, tab.end(), _TABULAR_ARGS[kind])
            stop = block.find(f"\\end{{{kind}}}", start)
            rows.extend(parse_tabular_rows(block[start: stop if stop != -1 else len(block)]))
        if rows:
            tables.append({"caption": caption, "rows": rows})
    return tables


def table_to_markdown(rows: List[List[str]], max_rows: int = 80) -> str:
    """
    Markdown table: header row, separator row, data rows (the contract
    QasperChunker.split_table relies on). Very long tables are truncated.
    """
    rows = rows[:max_rows]
    width = max(len(r) for r in rows)
    norm = [[c.replace("|", "/") for c in r] + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(norm[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in norm[1:]]
    return "\n".join(lines)


# ── Matching LaTeX tables to QASPER captions ──────────────────────────────────

_LABEL_PREFIX_RE = re.compile(r"^\s*(table|figure|fig\.?)\s*\d+\s*[:.]?\s*", re.IGNORECASE)


def _caption_tokens(caption: str) -> set:
    caption = _LABEL_PREFIX_RE.sub("", caption.lower())
    return set(re.findall(r"[a-z0-9]+", caption))


def caption_similarity(a: str, b: str) -> float:
    ta, tb = _caption_tokens(a), _caption_tokens(b)
    if not ta or not tb:
        return 0.0
    return 2 * len(ta & tb) / (len(ta) + len(tb))


def match_tables(qasper_captions: List[str], latex_tables: List[Dict],
                 min_similarity: float = 0.6) -> Dict[str, str]:
    """
    Greedy one-to-one matching of QASPER table captions to LaTeX tables by
    caption token overlap (Dice). Returns {qasper_caption: markdown_body}.
    """
    pairs = sorted(
        ((caption_similarity(qc, lt["caption"]), qi, li)
         for qi, qc in enumerate(qasper_captions)
         for li, lt in enumerate(latex_tables)),
        reverse=True,
    )
    used_q, used_l, matched = set(), set(), {}
    for score, qi, li in pairs:
        if score < min_similarity:
            break
        if qi in used_q or li in used_l:
            continue
        used_q.add(qi)
        used_l.add(li)
        matched[qasper_captions[qi].strip()] = table_to_markdown(latex_tables[li]["rows"])
    return matched


def build_table_bodies(papers, cache_dir: Path, delay: float = 3.0,
                       limit: Optional[int] = None) -> Dict[str, Dict[str, str]]:
    bodies: Dict[str, Dict[str, str]] = {}
    n_src = n_table_captions = n_matched = 0
    for i, paper in enumerate(papers):
        if limit is not None and i >= limit:
            break
        captions = (paper.get("figures_and_tables") or {}).get("caption") or []
        table_captions = [c for c in captions
                          if c and re.match(r"^\s*table", c, re.IGNORECASE)]
        n_table_captions += len(table_captions)
        if not table_captions:
            continue
        tex = fetch_tex(paper["id"], cache_dir, delay=delay)
        if not tex:
            continue
        n_src += 1
        matched = match_tables(table_captions, extract_tables(tex))
        if matched:
            bodies[paper["id"]] = matched
            n_matched += len(matched)
        if (i + 1) % 25 == 0:
            logger.info("[%d papers] sources=%d table captions=%d matched=%d",
                        i + 1, n_src, n_table_captions, n_matched)
    logger.info("Done: %d papers with LaTeX source; %d/%d table captions matched to a body.",
                n_src, n_matched, n_table_captions)
    return bodies


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Extract QASPER table bodies from arXiv LaTeX.")
    parser.add_argument("--out", default=str(_PROJECT_ROOT / "data" / "table_bodies.json"))
    parser.add_argument("--cache-dir", default=str(_PROJECT_ROOT / "data" / "arxiv_src"))
    parser.add_argument("--delay", type=float, default=3.0,
                        help="Seconds between arXiv requests (cache misses only).")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N papers.")
    args = parser.parse_args()

    from src.data.make_dataset import load_and_inspect_qasper
    papers = load_and_inspect_qasper()
    bodies = build_table_bodies(papers, Path(args.cache_dir), delay=args.delay, limit=args.limit)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(bodies, fh, indent=1)
    logger.info("Wrote %s (%d papers).", args.out, len(bodies))


if __name__ == "__main__":
    main()

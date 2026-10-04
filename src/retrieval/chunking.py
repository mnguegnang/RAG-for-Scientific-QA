import re
from transformers import AutoTokenizer
from typing import Dict, List, Optional

# "Table 3: ...", "Figure 2. ...", "Fig. 4: ..." at the start of a QASPER caption.
_FLOAT_LABEL_RE = re.compile(r'^\s*(Table|Figure|Fig\.?)\s*(\d+)', re.IGNORECASE)
# QASPER figure files look like "4-Table1-1.png": the leading number is the page.
_FLOAT_PAGE_RE = re.compile(r'^(\d+)-')

TABLE_FORMATS = ("markdown", "rows")


def _markdown_cells(line: str) -> List[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def linearize_markdown_table(body: str) -> List[str]:
    """
    One natural-language sentence per data row, read horizontally:
    "Row 3: Methods is our; AIDA-B is 94.3%."

    This is TabFact's "horizontal template" linearization (Chen et al.,
    ICLR 2020, §3.2): with cells joined by copulas and punctuation, a
    text-pretrained entailment model verified table statements at 65.1%
    test accuracy, against 50.4% (chance level) for the same table as plain
    concatenated cells (Table 2). The support checkers here (MiniCheck,
    Prometheus) and the generator are text-pretrained as well.

    *body* follows the arxiv_tables.table_to_markdown contract (header row,
    separator row, data rows). Empty cells are skipped; an empty header
    becomes "column N".
    """
    lines = [l for l in body.split("\n") if l.strip()]
    if len(lines) < 3:
        return []
    header = _markdown_cells(lines[0])
    sentences = []
    for i, line in enumerate(lines[2:], 1):
        cells = _markdown_cells(line)
        parts = [f"{header[j] if j < len(header) and header[j] else f'column {j + 1}'} is {cell}"
                 for j, cell in enumerate(cells) if cell]
        if parts:
            sentences.append(f"Row {i}: " + "; ".join(parts) + ".")
    return sentences


class QasperChunker:
    def __init__(self, model_name="allenai/specter2_base", max_tokens=500, overlap_pct=0.1,
                 table_format: str = "markdown"):
        """
        Initializes the chunker.

        CRITICAL SOURCING:
        1. model_name: MUST be "allenai/specter2_base" to match your DenseIndexer.
           Chunking must be done using the bottleneck model's tokenizer.
        2. max_tokens: Set to 500 (leaving 12 tokens for [CLS] and [SEP] special tokens
           required by SPECTER2's 512 absolute limit).
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Prevent HuggingFace from throwing console warnings when calculating length
        self.tokenizer.model_max_length = int(1e30)

        self.max_tokens = max_tokens
        # How table bodies are written into chunks: "markdown" (header, separator
        # and data rows) or "rows" (one linearized sentence per data row).
        if table_format not in TABLE_FORMATS:
            raise ValueError(f"table_format must be one of {TABLE_FORMATS}")
        self.table_format = table_format
        # Calculate overlap tokens (e.g., 500 * 0.1 = 50 tokens)
        self.overlap_tokens = int(max_tokens * overlap_pct)

    def n_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def split_large_text(self, text: str) -> List[str]:
        """
        Splits a single long text string into overlapping chunks based on token count.
        """
        # add_special_tokens=False because the embedding model will add [CLS]/[SEP] during the encode step
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        # Case 1: Text fits in one chunk (common for academic paragraphs)
        if len(tokens) <= self.max_tokens:
            return [text]

        # Case 2: Sliding Window for oversized paragraphs
        chunks = []
        stride = self.max_tokens - self.overlap_tokens

        if stride <= 0:
            raise ValueError("Overlap percentage is too high, stride must be > 0")

        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i : i + self.max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

        return chunks

    def split_table(self, caption: str, body: str) -> List[str]:
        """
        Caption + table body, split row-wise so every chunk repeats the
        caption (and, for Markdown, the header row) and stays within
        max_tokens. A table split mid-row would leave numbers without the
        column names that give them meaning, so the sliding window used for
        prose is not used here.

        *body* follows the arxiv_tables.table_to_markdown contract: header
        row, separator row, then data rows. With table_format="rows" each
        data row is first linearized into a sentence that names its columns.
        """
        if body and self.table_format == "rows":
            rows = linearize_markdown_table(body)
            prefix = caption
        else:
            rows = body.split("\n")[2:] if body else []
            prefix = f"{caption}\n" + "\n".join(body.split("\n")[:2]) if body else caption
        text = f"{prefix}\n" + "\n".join(rows) if rows else caption
        if not rows or self.n_tokens(text) <= self.max_tokens:
            return [text]

        budget = self.max_tokens - self.n_tokens(prefix) - 8
        chunks, current, used = [], [], 0
        for row in rows:
            cost = self.n_tokens(row) + 1
            if current and used + cost > budget:
                chunks.append(f"{prefix}\n" + "\n".join(current))
                current, used = [], 0
            current.append(row)
            used += cost
        if current:
            chunks.append(f"{prefix}\n" + "\n".join(current))
        # A single oversized row still has to fit the encoder.
        return [piece for chunk in chunks for piece in self.split_large_text(chunk)]

    @staticmethod
    def _float_section(caption: str) -> str:
        m = _FLOAT_LABEL_RE.match(caption or "")
        if not m:
            return "Tables and Figures"
        kind = "Figure" if m.group(1).lower().startswith("fig") else "Table"
        return f"{kind} {m.group(2)}"

    @staticmethod
    def _float_type(caption: str, file_name: str) -> str:
        m = _FLOAT_LABEL_RE.match(caption or "")
        if m:
            return "figure" if m.group(1).lower().startswith("fig") else "table"
        return "table" if "table" in (file_name or "").lower() else "figure"

    def process_paper(self, paper_data: Dict,
                      table_bodies: Optional[Dict[str, str]] = None) -> List[Dict]:
        """
        Flattens a QASPER paper into a list of chunk dictionaries with metadata.

        Chunks, in reading order (``position``):
          1. the abstract (chunk_type "abstract");
          2. every full-text paragraph (chunk_type "text");
          3. every table / figure caption (chunk_type "table" / "figure"),
             ordered by page. When *table_bodies* maps a caption to an
             extracted Markdown table (src/data/arxiv_tables.py), the body is
             appended so the numbers themselves become retrievable.

        Tables and figures: 60 of the 150 evaluation questions have
        "FLOAT SELECTED" (table/figure) gold evidence, which the text-only index
        could never retrieve (key_metrics_improvements.md, 2026-10-03, D1/P2;
        Dasigi et al. 2021 §1 and Table 1).
        """
        chunks = []
        paper_id = paper_data['id']
        title = paper_data['title']
        table_bodies = table_bodies or {}
        position = 0

        def add(text: str, section_name: str, para_id: str, chunk_type: str):
            nonlocal position
            # Anthropic (2024) Contextual Retrieval — prepend document-level
            # labels so SPECTER2 encodes *which* paper and section each passage
            # belongs to. This anchors the chunk vector in a paper-specific
            # region of the embedding space rather than a generic topic space.
            pieces = (self.split_table(text, table_bodies.get(text, ""))
                      if chunk_type == "table" else self.split_large_text(text))
            for sub_idx, piece in enumerate(pieces):
                chunks.append({
                    "paper_id": paper_id,
                    "title": title,
                    "section_name": section_name,
                    "original_para_id": para_id,
                    "chunk_id": f"{para_id}_{sub_idx}",
                    "chunk_type": chunk_type,
                    "position": position,
                    "text": (
                        f"Title: {title or 'Unknown'}. "
                        f"Section: {section_name or 'Unknown'}.\n"
                        + piece
                    ),
                })
                position += 1

        abstract = (paper_data.get('abstract') or '').strip()
        if abstract:
            add(abstract, "Abstract", f"{paper_id}_abstract", "abstract")

        sections = paper_data['full_text']['section_name']
        paragraphs_list = paper_data['full_text']['paragraphs']
        for section_idx, (section_name, section_paras) in enumerate(zip(sections, paragraphs_list)):
            for para_idx, paragraph in enumerate(section_paras):
                if paragraph and paragraph.strip():
                    add(paragraph, section_name, f"{paper_id}_{section_idx}_{para_idx}", "text")

        floats = paper_data.get('figures_and_tables') or {}
        captions = floats.get('caption') or []
        files = floats.get('file') or [""] * len(captions)
        order = sorted(
            range(len(captions)),
            key=lambda i: (int(m.group(1)) if (m := _FLOAT_PAGE_RE.match(files[i] or "")) else 10**6, i),
        )
        for float_idx in order:
            caption = (captions[float_idx] or "").strip()
            if not caption:
                continue
            add(caption, self._float_section(caption), f"{paper_id}_float_{float_idx}",
                self._float_type(caption, files[float_idx]))

        return chunks

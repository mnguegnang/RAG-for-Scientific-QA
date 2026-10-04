import pytest

try:
    from src.retrieval.chunking import QasperChunker
    CHUNKER = QasperChunker()          # needs the cached allenai/specter2_base tokenizer
except Exception as exc:               # pragma: no cover - environment without the model
    pytest.skip(f"SPECTER2 tokenizer unavailable: {exc}", allow_module_level=True)

PAPER = {
    "id": "0000.00001",
    "title": "A Paper",
    "abstract": "We study things.",
    "full_text": {"section_name": ["Intro", "Results"],
                  "paragraphs": [["First paragraph.", ""], ["Results paragraph."]]},
    "figures_and_tables": {
        "caption": ["Table 2: Main results.", "Figure 1: Overview of the model.",
                    "Table 1: Dataset statistics."],
        "file": ["5-Table2-1.png", "2-Figure1-1.png", "3-Table1-1.png"],
    },
}


def test_reading_order_types_and_positions():
    chunks = CHUNKER.process_paper(PAPER)
    assert [c["chunk_type"] for c in chunks] == ["abstract", "text", "text", "figure", "table", "table"]
    assert [c["position"] for c in chunks] == list(range(len(chunks)))
    # floats ordered by page: Figure 1 (p2), Table 1 (p3), Table 2 (p5)
    assert [c["section_name"] for c in chunks[3:]] == ["Figure 1", "Table 1", "Table 2"]
    assert chunks[0]["text"] == "Title: A Paper. Section: Abstract.\nWe study things."
    assert chunks[1]["chunk_id"] == "0000.00001_0_0_0"          # legacy id scheme kept
    assert all(c["text"].strip() for c in chunks)                # empty paragraph skipped


def test_table_body_appended_and_split_with_repeated_header():
    body = "| Model | Acc |\n|---|---|\n" + "\n".join(f"| model{i} | 0.{i:03d} |" for i in range(400))
    chunks = CHUNKER.process_paper(PAPER, table_bodies={"Table 2: Main results.": body})
    table2 = [c for c in chunks if c["section_name"] == "Table 2"]
    assert len(table2) > 1
    for c in table2:
        lines = c["text"].split("\n")
        assert lines[1] == "Table 2: Main results."
        assert lines[2:4] == ["| Model | Acc |", "|---|---|"]
        assert CHUNKER.n_tokens(c["text"]) <= CHUNKER.max_tokens + 20
    rows = [l for c in table2 for l in c["text"].split("\n")[4:]]
    assert len(rows) == 400                                       # no row lost or duplicated

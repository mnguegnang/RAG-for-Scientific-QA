import gzip
import io
import tarfile

from src.data.arxiv_tables import (
    caption_similarity, extract_tables, extract_tex, latex_to_text, match_tables,
    table_to_markdown,
)

TEX = r"""
\begin{table*}[t]
\centering
% \caption{An old commented-out caption}
\begin{tabular*}{\textwidth}{l|cc}
\toprule
\textbf{Model} & \multicolumn{2}{c}{F$_1$ (\%)} \\
\midrule
BERT~\cite{devlin} & 84.3 & $\pm$0.2 \\ \cline{2-3}
{\bf Ours} & \textbf{86.1} & 0.1 \\[2pt]
\bottomrule
\end{tabular*}
\caption{Results on the {\sc SQuAD} dev set. Best in \textbf{bold}.}
\label{tab:main}
\end{table*}
\begin{figure}\caption{Not a table}\end{figure}
"""


def test_latex_to_text_handles_common_markup():
    assert latex_to_text(r"\textbf{86.1}") == "86.1"
    assert latex_to_text(r"\multicolumn{2}{c}{F$_1$ (\%)}") == "F_1 (%)"
    assert latex_to_text(r"BERT~\cite{devlin}") == "BERT"
    assert latex_to_text(r"$\pm$0.2") == "±0.2"


def test_extract_tables_reads_caption_and_rows():
    tables = extract_tables(TEX)
    assert len(tables) == 1
    t = tables[0]
    assert t["caption"] == "Results on the SQuAD dev set. Best in bold."
    assert t["rows"] == [["Model", "F_1 (%)"], ["BERT", "84.3", "±0.2"], ["Ours", "86.1", "0.1"]]


def test_markdown_contract_header_separator_rows():
    md = table_to_markdown([["Model", "F1"], ["BERT", "84.3", "x"]]).split("\n")
    assert md[0] == "| Model | F1 |  |"
    assert md[1] == "|---|---|---|"
    assert md[2] == "| BERT | 84.3 | x |"


def test_match_tables_by_caption_ignoring_label_prefix():
    tables = extract_tables(TEX)
    matched = match_tables(["Table 2: Results on the SQuAD dev set. Best in bold.",
                            "Table 3: Something else entirely."], tables)
    assert list(matched) == ["Table 2: Results on the SQuAD dev set. Best in bold."]
    assert "| Ours | 86.1 | 0.1 |" in next(iter(matched.values()))
    assert caption_similarity("Table 1: a b c", "a b c") == 1.0


def test_extract_tex_from_tarball_and_single_gzip():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in (("main.tex", b"\\documentclass{article}"), ("fig.png", b"\x89PNG")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    assert extract_tex(buf.getvalue()) == "\\documentclass{article}"
    assert extract_tex(gzip.compress(b"\\begin{document}x")) == "\\begin{document}x"
    assert extract_tex(b"%PDF-1.5 not a source") == ""

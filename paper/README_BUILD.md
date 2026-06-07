# Building the paper

`main.tex` is the consolidated CCF-A conference paper (ACM `sigconf` format — the
template used by KDD / CIKM / WWW / SIGIR, all CCF A). It integrates the full research
program: contributions C1–C6, the SOTA benchmark (§6), the LLM architectural-limits
framing, and all tables + figures.

## Compile

**Overleaf (easiest):** upload `main.tex`, `references.bib`, and `sections/figures/`,
set compiler to pdfLaTeX, compile. `acmart.cls` is built in.

**Local (TeX Live / MiKTeX):**
```
cd paper
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```
Requires `acmart.cls` (TeX Live full, or `tlmgr install acmart`).

## Notes
- Figures are read from `sections/figures/` via `\graphicspath`. All 8 referenced PNGs
  are present and committed.
- `nonacm` + `printacmref=false` suppress the ACM copyright block (remove for a real
  ACM submission and add the rights commands the venue provides).
- To retarget **AAAI** (also CCF A): swap the preamble for `aaai24.sty`/`aaai25.sty`,
  move the abstract/keywords into the AAAI macros, and switch the bib style; the body,
  tables, and figures carry over unchanged.
- The 28 `\cite` keys all resolve against `references.bib` (46 entries); the 8
  `\includegraphics` paths all resolve. Verified statically (no LaTeX toolchain in the
  dev environment).
- `references.bib` entry `regimeawarelgbm2026` (MDPI) carries a `note` flag: its author
  list must be completed from the publisher page before submission.

## Source-of-truth sections
The long-form Markdown sections under `sections/` remain the editable source; `main.tex`
is the assembled, submission-formatted consolidation. Figures are regenerable via the
`src/**/summarize_*.py` and `src/figviz/methods_tree.py` scripts.

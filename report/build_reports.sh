#!/usr/bin/env bash
# Build the PDF of every report from its LaTeX source. Needs XeLaTeX (tectonic or TeX Live) and, for the Hindi
# examples in Question 1, a Devanagari font such as Noto (sudo apt-get install fonts-noto-core).
set -euo pipefail
cd "$(dirname "$0")/.."
for tex in Question-1/report/report.tex Question-2/report/report.tex Question-3/report/report.tex \
           Question-4/report/report.tex report/combined_report.tex; do
  dir=$(dirname "$tex"); file=$(basename "$tex")
  if command -v tectonic >/dev/null; then (cd "$dir" && tectonic "$file")
  else (cd "$dir" && latexmk -xelatex -interaction=nonstopmode "$file"); fi
  echo "built ${tex%.tex}.pdf"
done

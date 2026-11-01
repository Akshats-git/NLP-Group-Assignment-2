#!/usr/bin/env bash
# Build the submission ZIP from tracked files only, so datasets, .venv and untracked checkpoints are never included.
# Q4 checkpoints are tracked on purpose (the Q4 brief requires them).
set -euo pipefail
cd "$(dirname "$0")"
out="${1:-NLP_Group_Assignment_2_submission.zip}"
git archive --format=zip --prefix=NLP-Group-Assignment-2/ -o "$out" HEAD
echo "wrote $out ($(du -h "$out" | cut -f1))"

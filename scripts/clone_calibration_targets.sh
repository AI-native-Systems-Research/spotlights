#!/usr/bin/env bash
# Clone each historical Spotlights target repo at the exact commit its run was
# pinned to, so structural features (files, LOC) are measured against the tree
# the run actually saw -- not today's main.
#
# Fetches only the single pinned commit at depth 1 (GitHub serves arbitrary
# SHAs), which keeps multi-GB histories like vllm/rocksdb off the disk.
#
# Usage: bash scripts/clone_calibration_targets.sh /c/temp/spotlights-calib
set -u

DEST="${1:?usage: $0 <dest-dir>}"
mkdir -p "$DEST"

# name|url|pinned-sha   (sha from each run_manifest.json target.commit_sha)
TARGETS='
colpali|https://github.com/illuin-tech/colpali.git|b34c38815
vllm|https://github.com/vllm-project/vllm.git|bcf2be961
llm-d-router|https://github.com/llm-d/llm-d-router|5f4e762f3
py-inference-scheduler|https://github.com/llm-d-incubation/py-inference-scheduler|e760a6307
llm-d-workload-variant-autoscaler|https://github.com/llm-d/llm-d-workload-variant-autoscaler|cc0ce3ed7
rocksdb|https://github.com/facebook/rocksdb.git|abeebd963
'

fail=0
for row in $TARGETS; do
  name="${row%%|*}"; rest="${row#*|}"
  url="${rest%%|*}"; sha="${rest##*|}"
  dir="$DEST/$name"

  if [ -d "$dir/.git" ] && git -C "$dir" cat-file -t "$sha" >/dev/null 2>&1; then
    echo "== $name: already present at $sha, skipping"
    continue
  fi

  echo "== $name  <- $url @ $sha"
  rm -rf "$dir"
  git init -q "$dir" 2>/dev/null
  git -C "$dir" remote add origin "$url"

  # Preferred: single-commit shallow fetch.
  if git -C "$dir" fetch -q --depth 1 origin "$sha" 2>/dev/null; then
    git -C "$dir" checkout -q FETCH_HEAD
  else
    # Fallback: server refused a bare-SHA want -- take blobless full history.
    echo "   (bare-SHA fetch refused; falling back to blobless full fetch)"
    if git -C "$dir" fetch -q --filter=blob:none origin 2>/dev/null &&
       git -C "$dir" checkout -q "$sha" 2>/dev/null; then
      :
    else
      echo "   FAILED to obtain $sha"
      fail=$((fail + 1))
      continue
    fi
  fi
  echo "   ok -> $(git -C "$dir" rev-parse --short HEAD)"
done

echo
echo "clones in $DEST:"
for d in "$DEST"/*/; do
  [ -d "$d/.git" ] || continue
  printf '  %-36s %s\n' "$(basename "$d")" "$(git -C "$d" rev-parse --short HEAD 2>/dev/null)"
done
[ "$fail" -eq 0 ] || echo "WARNING: $fail target(s) failed"

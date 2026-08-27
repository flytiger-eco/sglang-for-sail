#!/bin/bash
# Warm the NFS page cache for the given model checkpoint directories by
# reading all safetensors files with bounded parallelism.
#
# Each nightly partition calls this with only the models its own suite
# actually loads, so cold-cache reads no longer pay for models owned by
# other partitions.
#
# Usage: warm_page_cache.sh <model_dir> [<model_dir> ...]
set -u

if [ "$#" -eq 0 ]; then
  echo "No model directories given; skipping warm-up."
  exit 0
fi

WARM_PARALLELISM="${WARM_PARALLELISM:-8}"

for dir in "$@"; do
  if [ ! -d "$dir" ]; then
    echo "WARN: $dir not found on this runner; skipping"
    continue
  fi
  echo "Pre-reading $(basename "$dir") into page cache..."
  find "$dir" -name "*.safetensors" -print0 2>/dev/null \
    | xargs -0 -r -P "$WARM_PARALLELISM" -n 1 cat > /dev/null
done

echo "Page cache warm-up done."

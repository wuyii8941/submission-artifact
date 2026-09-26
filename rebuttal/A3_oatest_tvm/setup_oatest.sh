#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OATEST_SRC=${OATEST_SRC:-$HERE/oatest}
OATEST_PATCHED=${OATEST_PATCHED:-$HERE/oatest_patched}
if [ ! -d "$OATEST_SRC/TVM" ]; then
  git clone https://github.com/ShenQingchao/OATest "$OATEST_SRC"
  git -C "$OATEST_SRC" checkout 7c28fe9f71
fi
find "$OATEST_SRC/res/tvm_ut" -name '*.py' -print0 | while IFS= read -r -d '' f; do
  if [ -n "$(tail -c1 "$f")" ]; then echo >> "$f"; fi
done
(cd "$OATEST_SRC" && sha256sum --quiet -c "$HERE/oatest_corpus.sha256")
echo "corpus: 942 files match"
rm -rf "$OATEST_PATCHED"
cp -r "$OATEST_SRC/TVM" "$OATEST_PATCHED"
patch --no-backup-if-mismatch -p2 -d "$OATEST_PATCHED" < "$HERE/oatest_splice_graph.patch"
echo "OATEST_SRC=$OATEST_SRC OATEST_PATCHED=$OATEST_PATCHED"

#!/usr/bin/env bash
set -euo pipefail

# Build a self-contained source package. The checked-in package uses the
# repository core path for development; release artifacts embed that core.
root=$(cd "$(dirname "$0")/.." && pwd)
out=${1:-"$root/dist"}
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
mkdir "$stage/mltreelib"
(
  cd "$root/r/mltreelib"
  tar --exclude='./src/target' --exclude='./src/Cargo.lock' \
    --exclude='./src/symbols.rds' --exclude='./src/*.o' \
    --exclude='./src/*.so' --exclude='./src/*.a' \
    --exclude='./tools/rust/target' --exclude='./tools/rust/Cargo.lock' \
    -cf - .
) | tar -xf - -C "$stage/mltreelib"
cp -a "$root/crates/mltreelib-core" "$stage/mltreelib/tools/rust/core"
sed -i 's#../../../../crates/mltreelib-core#core#' "$stage/mltreelib/tools/rust/Cargo.toml"
# The repository core inherits metadata from the root workspace. A source
# package has no root workspace, so write only the standalone package header.
awk 'BEGIN { in_deps=0 } /^\[dependencies\]/ { in_deps=1 } in_deps { print }' \
  "$stage/mltreelib/tools/rust/core/Cargo.toml" > "$stage/mltreelib/tools/rust/core/Cargo.toml.tmp"
cat > "$stage/mltreelib/tools/rust/core/Cargo.toml" <<'EOF'
[package]
name = "mltreelib-core"
version = "0.4.12"
edition = "2021"
license = "Apache-2.0"

EOF
cat "$stage/mltreelib/tools/rust/core/Cargo.toml.tmp" >> "$stage/mltreelib/tools/rust/core/Cargo.toml"
rm "$stage/mltreelib/tools/rust/core/Cargo.toml.tmp"
mkdir -p "$out"
(cd "$stage" && R CMD build mltreelib --no-manual --no-resave-data --no-build-vignettes)
mv "$stage"/mltreelib_*.tar.gz "$out/"

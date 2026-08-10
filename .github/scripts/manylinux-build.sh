#!/usr/bin/env bash
#
# Compiles the Statelet server binaries inside a manylinux image. Invoked by
# .github/workflows/publish-statelet-pypi.yml as
#
#   docker run ... "$BUILD_IMAGE" bash /work/lme/.github/scripts/manylinux-build.sh
#
# A file rather than an inline `bash -c '...'`: the inline form put the whole
# script inside single quotes, so one apostrophe in a comment ended the string
# and silently ran the remainder on the host instead of in the container.
#
# Expects, from `docker run -e`:
#   TARGET      rust target triple to build
#   CARGO_HOME  writable, on the mounted workspace so rustup survives the step
# and the engine checkout mounted at the working directory.
set -euo pipefail

: "${TARGET:?TARGET must be set}"
: "${CARGO_HOME:?CARGO_HOME must be set}"

echo "::group::Rust toolchain"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
  | sh -s -- -y --profile minimal --default-toolchain stable --target "$TARGET"
export PATH="$CARGO_HOME/bin:$PATH"
cargo --version
rustc --version
echo "::endgroup::"

echo "::group::libclang for bindgen"
# librocksdb-sys drives bindgen, which needs a real libclang.so. Note that
# manylinux-install-clang is no help: it installs a *static* toolchain —
# /opt/clang/bin/clang and friends, no shared library at all.
#
# The SCL llvm-toolset carries a usable libclang together with its resource
# headers, and this image already has those repos configured because its
# devtoolset gcc comes from them. CentOS 7's base clang is 3.4, far below what
# bindgen accepts, so it is not a candidate.
find_libclang() { find /usr /opt -name 'libclang.so*' 2>/dev/null | head -1; }

LIBCLANG="$(find_libclang)"
if [ -z "$LIBCLANG" ]; then
    yum -y install llvm-toolset-7.0-clang >/dev/null || true
    LIBCLANG="$(find_libclang)"
fi
if [ -z "$LIBCLANG" ]; then
    echo "::error::no libclang.so in the image; bindgen cannot run"
    find / -xdev -maxdepth 6 -name '*clang*' 2>/dev/null | head -40
    exit 1
fi
export LIBCLANG_PATH="${LIBCLANG%/*}"
echo "libclang: $LIBCLANG"
echo "::endgroup::"

echo "::group::static OpenSSL"
# The engine asks for openssl with `vendored`, but that is not the copy that
# matters here. ort's default features include tls-native, which drags
# ort-sys -> ureq -> native-tls -> openssl-sys into the build-script graph, and
# under edition 2024 the resolver keeps those features separate from the
# ordinary dependency graph. That copy resolves to plain `default`, has no
# vendored path compiled into it, and goes looking for a system OpenSSL;
# `--features openssl/vendored` cannot reach it.
#
# macOS never hits this because native-tls uses Security.framework there, and a
# bare ubuntu runner never hits it because libssl-dev is already installed.
# Only a clean container exposes it.
#
# Static, and built here, for two reasons: libssl is not on the manylinux
# allowlist, so a dynamic link would leave the wheel depending on a library the
# installing machine may not have; and CentOS 7 ships OpenSSL 1.0.2, which
# openssl-sys 0.9 no longer supports at all.
#
# The tidier fix belongs in the engine — ort with default-features = false
# would drop ureq, native-tls and this second openssl-sys entirely, since
# load-dynamic makes the binary download pointless — but that is a manifest
# change to make deliberately, not a side effect of a packaging workflow.
OPENSSL_VERSION=3.0.16
yum -y install perl-core perl-IPC-Cmd >/dev/null
curl -fsSL --retry 5 -o /tmp/openssl.tar.gz \
    "https://github.com/openssl/openssl/releases/download/openssl-${OPENSSL_VERSION}/openssl-${OPENSSL_VERSION}.tar.gz"
mkdir -p /tmp/openssl-src
tar -C /tmp/openssl-src --strip-components=1 -xzf /tmp/openssl.tar.gz
(
    cd /tmp/openssl-src
    ./Configure --prefix=/opt/openssl-static --openssldir=/opt/openssl-static \
        no-shared no-tests >/dev/null
    make -j"$(nproc)" >/dev/null
    make install_sw >/dev/null
)
export OPENSSL_DIR=/opt/openssl-static
export OPENSSL_STATIC=1
"$OPENSSL_DIR/bin/openssl" version
echo "::endgroup::"

echo "::group::cargo build"
# --locked: build exactly the graph in the engine's committed Cargo.lock, so a
# floating dependency cannot pick up an untested release (ort rc.12 -> rc.13
# broke this once) and end up inside a published wheel.
cargo build --locked --release \
    --features data-node,agent-state \
    --target "$TARGET"
echo "::endgroup::"

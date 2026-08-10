#!/usr/bin/env bash
#
# Builds one statelet .deb from an unpacked stage directory.
#
# Env:
#   VERSION    package version, no leading v
#   ARCH       debian architecture (amd64 | arm64)
#   STAGE_DIR  directory holding the binaries
#   UNIT_FILE  path to the systemd unit
#   LICENSE_FILE  path to LICENSE
#   GLIBC_MIN  minimum glibc the binaries need, e.g. 2.17
set -euo pipefail

: "${VERSION:?}" "${ARCH:?}" "${STAGE_DIR:?}" "${UNIT_FILE:?}" "${GLIBC_MIN:?}"
LICENSE_FILE="${LICENSE_FILE:-}"

PKG="statelet_${VERSION}_${ARCH}"
rm -rf "$PKG"
mkdir -p "$PKG/DEBIAN" "$PKG/usr/bin" "$PKG/usr/lib/systemd/system" \
         "$PKG/var/lib/statelet" "$PKG/usr/share/doc/statelet"

cp "$STAGE_DIR"/statelet-* "$PKG/usr/bin/"
chmod 755 "$PKG"/usr/bin/statelet-*
cp "$UNIT_FILE" "$PKG/usr/lib/systemd/system/"
if [ -n "$LICENSE_FILE" ] && [ -f "$LICENSE_FILE" ]; then
    cp "$LICENSE_FILE" "$PKG/usr/share/doc/statelet/copyright"
else
    echo "Apache-2.0" > "$PKG/usr/share/doc/statelet/copyright"
fi

# `dpkg-deb --build` does not run dpkg-shlibdeps, so nothing derives the libc
# dependency for us — a control file with no Depends installs cheerfully onto a
# machine too old to run the binaries, and the failure only shows up when the
# service will not start. The floor is whatever the build image guarantees.
cat > "$PKG/DEBIAN/control" <<CTRL
Package: statelet
Version: ${VERSION}
Architecture: ${ARCH}
Maintainer: Statelet Team <team@statelet.ai>
Depends: libc6 (>= ${GLIBC_MIN}), libgcc-s1 | libgcc1, libstdc++6
Description: Distributed key-value storage engine with LSM-tree, Raft, and vector index
 Statelet is a distributed KV store in Rust with HNSW vector search,
 temporal graph engine, and Redis-compatible protocol support.
Homepage: https://github.com/stateletlab/statelet
Priority: optional
Section: database
CTRL

cat > "$PKG/DEBIAN/postinst" <<'POSTINST'
#!/bin/sh
set -e
systemctl daemon-reload
systemctl enable statelet.service
systemctl start statelet.service || true
POSTINST

cat > "$PKG/DEBIAN/prerm" <<'PRERM'
#!/bin/sh
set -e
systemctl stop statelet.service || true
systemctl disable statelet.service || true
PRERM

cat > "$PKG/DEBIAN/postrm" <<'POSTRM'
#!/bin/sh
set -e
if [ "$1" = "purge" ]; then
    rm -rf /var/lib/statelet
fi
systemctl daemon-reload || true
POSTRM

chmod 755 "$PKG/DEBIAN/postinst" "$PKG/DEBIAN/prerm" "$PKG/DEBIAN/postrm"

dpkg-deb --build --root-owner-group "$PKG"
mv "${PKG}.deb" "statelet_${VERSION}_${ARCH}.deb"

echo "--- built statelet_${VERSION}_${ARCH}.deb"
dpkg-deb --info "statelet_${VERSION}_${ARCH}.deb"
dpkg-deb --contents "statelet_${VERSION}_${ARCH}.deb"

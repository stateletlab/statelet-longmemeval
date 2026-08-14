#!/usr/bin/env bash
#
# Builds one .deb from an unpacked stage directory. The defaults produce the
# EE `statelet` package; the lite workflow overrides the PKG_* knobs to ship
# `statelet-lite` from the same script.
#
# Env:
#   VERSION    package version, no leading v
#   ARCH       debian architecture (amd64 | arm64)
#   STAGE_DIR  directory holding the binaries
#   UNIT_FILE  path to the systemd unit; its basename is the service the
#              package enables and starts
#   LICENSE_FILE  path to LICENSE
#   GLIBC_MIN  minimum glibc the binaries need, e.g. 2.17
# Optional:
#   PKG_NAME       package name                          (default: statelet)
#   PKG_SUMMARY    Description: synopsis line
#   PKG_DESC_LONG  Description: continuation text, may span lines
#   PKG_LICENSE    copyright fallback when LICENSE_FILE is absent
#   PKG_CONFLICTS  emitted as a Conflicts: field; omitted when empty
set -euo pipefail

: "${VERSION:?}" "${ARCH:?}" "${STAGE_DIR:?}" "${UNIT_FILE:?}" "${GLIBC_MIN:?}"
LICENSE_FILE="${LICENSE_FILE:-}"
PKG_NAME="${PKG_NAME:-statelet}"
PKG_SUMMARY="${PKG_SUMMARY:-Distributed key-value storage engine with LSM-tree, Raft, and vector index}"
PKG_DESC_LONG="${PKG_DESC_LONG:-Statelet is a distributed KV store in Rust with HNSW vector search,
temporal graph engine, and Redis-compatible protocol support.}"
PKG_LICENSE="${PKG_LICENSE:-Apache-2.0}"
PKG_CONFLICTS="${PKG_CONFLICTS:-}"
SERVICE="$(basename "$UNIT_FILE")"

PKG="${PKG_NAME}_${VERSION}_${ARCH}"
rm -rf "$PKG"
mkdir -p "$PKG/DEBIAN" "$PKG/usr/bin" "$PKG/usr/lib/systemd/system" \
         "$PKG/var/lib/${PKG_NAME}" "$PKG/usr/share/doc/${PKG_NAME}" \
         "$PKG/usr/share/statelet"

cp "$STAGE_DIR"/statelet-* "$PKG/usr/bin/"
chmod 755 "$PKG"/usr/bin/statelet-*
cp "$UNIT_FILE" "$PKG/usr/lib/systemd/system/"
# Every server binary — statelet-lite included — resolves the admin UI
# relative to its own executable: /usr/bin/<binary> -> /usr/share/statelet/ui.
# The path therefore does not vary with PKG_NAME; two packages that would both
# claim it must declare PKG_CONFLICTS instead.
if [ -d "$STAGE_DIR/ui" ]; then
    cp -R "$STAGE_DIR/ui" "$PKG/usr/share/statelet/ui"
else
    echo "warning: no admin UI in $STAGE_DIR; the management port will serve 404s" >&2
fi
if [ -n "$LICENSE_FILE" ] && [ -f "$LICENSE_FILE" ]; then
    cp "$LICENSE_FILE" "$PKG/usr/share/doc/${PKG_NAME}/copyright"
else
    echo "$PKG_LICENSE" > "$PKG/usr/share/doc/${PKG_NAME}/copyright"
fi

# `dpkg-deb --build` does not run dpkg-shlibdeps, so nothing derives the libc
# dependency for us — a control file with no Depends installs cheerfully onto a
# machine too old to run the binaries, and the failure only shows up when the
# service will not start. The floor is whatever the build image guarantees.
{
    cat <<CTRL
Package: ${PKG_NAME}
Version: ${VERSION}
Architecture: ${ARCH}
Maintainer: Statelet Team <team@statelet.ai>
Depends: libc6 (>= ${GLIBC_MIN}), libgcc-s1 | libgcc1, libstdc++6
Homepage: https://github.com/stateletlab/statelet
Priority: optional
Section: database
CTRL
    if [ -n "$PKG_CONFLICTS" ]; then
        printf 'Conflicts: %s\n' "$PKG_CONFLICTS"
    fi
    printf 'Description: %s\n' "$PKG_SUMMARY"
    # Continuation lines of the extended description carry one leading space.
    printf '%s\n' "$PKG_DESC_LONG" | sed 's/^/ /'
} > "$PKG/DEBIAN/control"

cat > "$PKG/DEBIAN/postinst" <<POSTINST
#!/bin/sh
set -e
systemctl daemon-reload
systemctl enable ${SERVICE}
systemctl start ${SERVICE} || true
POSTINST

cat > "$PKG/DEBIAN/prerm" <<PRERM
#!/bin/sh
set -e
systemctl stop ${SERVICE} || true
systemctl disable ${SERVICE} || true
PRERM

cat > "$PKG/DEBIAN/postrm" <<POSTRM
#!/bin/sh
set -e
if [ "\$1" = "purge" ]; then
    rm -rf /var/lib/${PKG_NAME}
fi
systemctl daemon-reload || true
POSTRM

chmod 755 "$PKG/DEBIAN/postinst" "$PKG/DEBIAN/prerm" "$PKG/DEBIAN/postrm"

# dpkg-deb names the output after the directory, which is already
# <pkg>_<version>_<arch> — the `mv` this used to carry moved the file onto
# itself and failed the build.
dpkg-deb --build --root-owner-group "$PKG"

echo "--- built ${PKG}.deb"
dpkg-deb --info "${PKG}.deb"
dpkg-deb --contents "${PKG}.deb"

#!/usr/bin/env bash
#
# Builds one .rpm from an unpacked stage directory. The defaults produce the
# EE `statelet` package; the lite workflow overrides the PKG_* knobs to ship
# `statelet-lite` from the same script.
#
# The payload is staged outside the buildroot and copied in by %install.
# Staging directly into BUILDROOT does not work: rpmbuild runs
# %__spec_install_pre before %install, which wipes the buildroot, so the files
# are gone by the time %files looks for them and every entry fails with
# "File not found".
#
# Unlike dpkg-deb, rpmbuild scans the payload and generates
# `libc.so.6(GLIBC_2.x)` requires by itself, so the glibc floor does not have to
# be declared here — it is read back out of the binaries instead.
#
# Env:
#   VERSION    package version, no leading v
#   ARCH       rpm architecture (x86_64 | aarch64)
#   STAGE_DIR  directory holding the binaries and the built admin UI
#   UNIT_FILE  path to the systemd unit; its basename is the service the
#              package enables and starts
# Optional:
#   PKG_NAME       package name                          (default: statelet)
#   PKG_SUMMARY    Summary: line
#   PKG_DESC_LONG  %description body, may span lines
#   PKG_LICENSE    License: tag                          (default: Apache-2.0)
#   PKG_CONFLICTS  emitted as a Conflicts: tag; omitted when empty
set -euo pipefail

: "${VERSION:?}" "${ARCH:?}" "${STAGE_DIR:?}" "${UNIT_FILE:?}"
PKG_NAME="${PKG_NAME:-statelet}"
PKG_SUMMARY="${PKG_SUMMARY:-Distributed key-value storage engine with LSM-tree, Raft, and vector index}"
PKG_DESC_LONG="${PKG_DESC_LONG:-Statelet is a distributed KV store in Rust with HNSW vector search,
temporal graph engine, and Redis-compatible protocol support.}"
PKG_LICENSE="${PKG_LICENSE:-Apache-2.0}"
PKG_CONFLICTS="${PKG_CONFLICTS:-}"
SERVICE="$(basename "$UNIT_FILE")"

TOP="$PWD/rpmbuild"
PAYLOAD="$PWD/rpm-payload"
rm -rf "$TOP" "$PAYLOAD"
mkdir -p "$TOP"/{BUILD,BUILDROOT,RPMS,SOURCES,SPECS,SRPMS}
mkdir -p "$PAYLOAD/usr/bin" "$PAYLOAD/usr/lib/systemd/system" \
         "$PAYLOAD/var/lib/${PKG_NAME}" "$PAYLOAD/usr/share/statelet"

cp "$STAGE_DIR"/statelet-* "$PAYLOAD/usr/bin/"
chmod 755 "$PAYLOAD"/usr/bin/statelet-*
cp "$UNIT_FILE" "$PAYLOAD/usr/lib/systemd/system/"
# Every server binary — statelet-lite included — resolves the admin UI as
# ../share/statelet/ui from its own location, so the path does not vary with
# PKG_NAME; two packages that would both claim it must declare PKG_CONFLICTS.
if [ -d "$STAGE_DIR/ui" ]; then
    cp -R "$STAGE_DIR/ui" "$PAYLOAD/usr/share/statelet/ui"
else
    echo "warning: no admin UI in $STAGE_DIR; the management port will serve 404s" >&2
fi

# The binary set differs per package (five for the cluster, one for lite), so
# the %files entries are read off the payload instead of being hardcoded.
BIN_FILES="$(cd "$PAYLOAD/usr/bin" && ls -1 | sed 's|^|/usr/bin/|')"

CONFLICTS_LINE=""
if [ -n "$PKG_CONFLICTS" ]; then
    CONFLICTS_LINE="Conflicts: ${PKG_CONFLICTS}"
fi

cat > "$TOP/SPECS/${PKG_NAME}.spec" <<SPEC
Name:    ${PKG_NAME}
Version: ${VERSION}
Release: 1
Summary: ${PKG_SUMMARY}
License: ${PKG_LICENSE}
URL:     https://github.com/stateletlab/statelet
${CONFLICTS_LINE}

# The payload is prebuilt; rpmbuild only assembles it.
%global debug_package %{nil}
%global __strip /bin/true

%description
${PKG_DESC_LONG}

%install
cp -a %{_payload}/. %{buildroot}/

%files
${BIN_FILES}
/usr/lib/systemd/system/${SERVICE}
/usr/share/statelet/ui
%dir /var/lib/${PKG_NAME}

%post
systemctl daemon-reload
systemctl enable ${SERVICE}
systemctl start ${SERVICE} || true

%preun
if [ "\$1" = 0 ]; then
    systemctl stop ${SERVICE} || true
    systemctl disable ${SERVICE} || true
fi

%postun
systemctl daemon-reload || true
if [ "\$1" = 0 ]; then
    rm -rf /var/lib/${PKG_NAME}
fi
SPEC

rpmbuild --define "_topdir $TOP" \
         --define "_payload $PAYLOAD" \
         --define "_arch ${ARCH}" \
         --target "${ARCH}" \
         -bb "$TOP/SPECS/${PKG_NAME}.spec"

cp "$TOP/RPMS/${ARCH}"/*.rpm .

RPM="${PKG_NAME}-${VERSION}-1.${ARCH}.rpm"
echo "--- built $RPM"
rpm -qip "$RPM"
echo "--- generated requires (the glibc floor should appear here):"
rpm -qpR "$RPM"

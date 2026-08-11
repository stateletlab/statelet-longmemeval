#!/usr/bin/env bash
#
# Builds one statelet .rpm from an unpacked stage directory.
#
# Unlike dpkg-deb, rpmbuild scans the payload and generates
# `libc.so.6(GLIBC_2.x)` requires by itself, so the glibc floor does not have to
# be declared here — it is read back out of the binaries instead.
#
# Env:
#   VERSION    package version, no leading v
#   ARCH       rpm architecture (x86_64 | aarch64)
#   STAGE_DIR  directory holding the binaries
#   UNIT_FILE  path to the systemd unit
set -euo pipefail

: "${VERSION:?}" "${ARCH:?}" "${STAGE_DIR:?}" "${UNIT_FILE:?}"

TOP="$PWD/rpmbuild"
BUILDROOT="$TOP/BUILDROOT/statelet-${VERSION}-1.${ARCH}"
rm -rf "$TOP"
mkdir -p "$TOP"/{BUILD,RPMS,SOURCES,SPECS,SRPMS}
mkdir -p "$BUILDROOT/usr/bin" "$BUILDROOT/usr/lib/systemd/system" \
         "$BUILDROOT/var/lib/statelet" "$BUILDROOT/usr/share/statelet"

cp "$STAGE_DIR"/statelet-* "$BUILDROOT/usr/bin/"
chmod 755 "$BUILDROOT"/usr/bin/statelet-*
cp "$UNIT_FILE" "$BUILDROOT/usr/lib/systemd/system/"
# /usr/bin/statelet-gateway resolves the UI as ../share/statelet/ui
if [ -d "$STAGE_DIR/ui" ]; then
    cp -R "$STAGE_DIR/ui" "$BUILDROOT/usr/share/statelet/ui"
else
    echo "warning: no admin UI in $STAGE_DIR; the management port will serve 404s" >&2
fi

cat > "$TOP/SPECS/statelet.spec" <<SPEC
Name:    statelet
Version: ${VERSION}
Release: 1
Summary: Distributed key-value storage engine with LSM-tree, Raft, and vector index
License: Apache-2.0
URL:     https://github.com/stateletlab/statelet

# The payload is prebuilt; rpmbuild only assembles it.
%global debug_package %{nil}
%global __strip /bin/true

%description
Statelet is a distributed KV store in Rust with HNSW vector search,
temporal graph engine, and Redis-compatible protocol support.

%install
# Files are already staged in BUILDROOT.

%files
/usr/bin/statelet-metadata
/usr/bin/statelet-datanode
/usr/bin/statelet-gateway
/usr/bin/statelet-cli
/usr/bin/statelet-cluster
/usr/lib/systemd/system/statelet.service
/usr/share/statelet/ui
%dir /var/lib/statelet

%post
systemctl daemon-reload
systemctl enable statelet.service
systemctl start statelet.service || true

%preun
if [ "\$1" = 0 ]; then
    systemctl stop statelet.service || true
    systemctl disable statelet.service || true
fi

%postun
systemctl daemon-reload || true
if [ "\$1" = 0 ]; then
    rm -rf /var/lib/statelet
fi
SPEC

rpmbuild --define "_topdir $TOP" \
         --define "_arch ${ARCH}" \
         --target "${ARCH}" \
         -bb "$TOP/SPECS/statelet.spec"

cp "$TOP/RPMS/${ARCH}"/*.rpm .

RPM="$(ls statelet-${VERSION}-1.${ARCH}.rpm)"
echo "--- built $RPM"
rpm -qip "$RPM"
echo "--- generated requires (glibc floor should appear here):"
rpm -qpR "$RPM"

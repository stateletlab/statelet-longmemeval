#!/usr/bin/env bash
#
# Builds one statelet .rpm from an unpacked stage directory.
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
#   UNIT_FILE  path to the systemd unit
set -euo pipefail

: "${VERSION:?}" "${ARCH:?}" "${STAGE_DIR:?}" "${UNIT_FILE:?}"

TOP="$PWD/rpmbuild"
PAYLOAD="$PWD/rpm-payload"
rm -rf "$TOP" "$PAYLOAD"
mkdir -p "$TOP"/{BUILD,BUILDROOT,RPMS,SOURCES,SPECS,SRPMS}
mkdir -p "$PAYLOAD/usr/bin" "$PAYLOAD/usr/lib/systemd/system" \
         "$PAYLOAD/var/lib/statelet" "$PAYLOAD/usr/share/statelet"

cp "$STAGE_DIR"/statelet-* "$PAYLOAD/usr/bin/"
chmod 755 "$PAYLOAD"/usr/bin/statelet-*
cp "$UNIT_FILE" "$PAYLOAD/usr/lib/systemd/system/"
# /usr/bin/statelet-gateway resolves the admin UI as ../share/statelet/ui
if [ -d "$STAGE_DIR/ui" ]; then
    cp -R "$STAGE_DIR/ui" "$PAYLOAD/usr/share/statelet/ui"
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
cp -a %{_payload}/. %{buildroot}/

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
         --define "_payload $PAYLOAD" \
         --define "_arch ${ARCH}" \
         --target "${ARCH}" \
         -bb "$TOP/SPECS/statelet.spec"

cp "$TOP/RPMS/${ARCH}"/*.rpm .

RPM="statelet-${VERSION}-1.${ARCH}.rpm"
echo "--- built $RPM"
rpm -qip "$RPM"
echo "--- generated requires (the glibc floor should appear here):"
rpm -qpR "$RPM"

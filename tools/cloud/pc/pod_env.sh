#!/bin/bash
# Sourced by the workstation-side scripts. Set from the "SSH over exposed TCP" line your cloud provider shows, e.g.
#   ssh USER@HOST -p PORT -i KEYFILE
# ->  export POD=USER@HOST PORT=PORT KEY=KEYFILE      (KEY optional; POD=local = dry run on this machine)
# POD_ROOT_REMOTE = $VC_ROOT on the pod (absolute path on the pod's persistent volume; required unless POD=local)
# POD_CODE_REMOTE = directory on the pod the code tarball is unpacked into (used by push_code.sh)
# POD_DATA_REMOTE = $VC_DATA on the pod (default $POD_ROOT_REMOTE/data)
: "${POD:?set POD=USER@HOST (from the SSH line of the cloud provider) or POD=local}"
: "${PORT:=22}"
KEYOPT=(); if [ -n "${KEY:-}" ]; then KEYOPT=(-i "$KEY"); fi
if [ "$POD" = local ]; then
  : "${POD_ROOT_REMOTE:=${VC_ROOT:-$HOME/vesuvius}}"
else
  : "${POD_ROOT_REMOTE:?set POD_ROOT_REMOTE=<absolute VC_ROOT on the pod>}"
fi
POD_DATA_REMOTE=${POD_DATA_REMOTE:-$POD_ROOT_REMOTE/data}
SSHOPT=(-p "$PORT" ${KEYOPT[@]+"${KEYOPT[@]}"} -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
if [ "$POD" = local ]; then
  rsh() { bash -c "$*"; }
  RSYNC_E=()
  RPATH() { echo "$1"; }
else
  rsh() { ssh "${SSHOPT[@]}" "$POD" "$*"; }
  RSYNC_E=(-e "ssh ${SSHOPT[*]}")
  RPATH() { echo "$POD:$1"; }
fi

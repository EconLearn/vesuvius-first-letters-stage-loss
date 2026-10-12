#!/bin/bash
# workstation -> pod: pack the repository's scripts (pack_code.sh, < 50 MB) and unpack them into $POD_CODE_REMOTE on the pod.
#   POD=USER@HOST PORT=NNNN [KEY=KEYFILE] POD_ROOT_REMOTE=/abs/vc_root POD_CODE_REMOTE=/abs/code bash tools/cloud/pc/push_code.sh
set -euo pipefail
. "$(dirname "$0")/pod_env.sh"
C=$(cd "$(dirname "$0")/.." && pwd)                                  # tools/cloud
DIST=${PACK_OUT:-${VC_WORK:-${VC_ROOT:-$HOME/vesuvius}/work}/cloud/dist}
bash "$C/pack_code.sh" "$DIST"
if [ "$POD" = local ]; then echo "POD=local: tarball at $DIST/vesuvius_code.tgz"; exit 0; fi
: "${POD_CODE_REMOTE:?set POD_CODE_REMOTE=<absolute directory on the pod to unpack the code into>}"
scp -P "$PORT" ${KEYOPT[@]+"${KEYOPT[@]}"} -o StrictHostKeyChecking=accept-new "$DIST/vesuvius_code.tgz" "$DIST/vesuvius_code.tgz.sha256" "$POD:"
rsh "cd \$HOME && [ \"\$(sha256sum vesuvius_code.tgz | cut -d' ' -f1)\" = \"\$(cat vesuvius_code.tgz.sha256)\" ] && mkdir -p $POD_CODE_REMOTE && tar -xzf vesuvius_code.tgz -C $POD_CODE_REMOTE && echo 'code unpacked into $POD_CODE_REMOTE'"

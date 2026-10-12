"""POD-ONLY disk-guard shim (installed by bootstrap.sh as sitecustomize.py into the pod's main venv; never use it on a
workstation).

Scripts outside this repository may check the free space of a fixed workstation path with shutil.disk_usage and a large
free-space floor. On a pod that path does not exist; what must be protected is the volume holding $VC_ROOT. When
$VC_ROOT/.POD exists and $POD_GUARD_PATH names such a path, a disk_usage query for that path (or anything below it)
returns the usage of the $VC_ROOT volume with POD_GUARD_OFFSET_GB (default 110) added to 'total' and 'free', so a guard
of the form "free >= floor + need" becomes "pod volume free >= (floor - offset) + need". Nothing else is touched, and
without $POD_GUARD_PATH nothing is changed at all. Scripts of this repository read tools/vcpaths.disk_guard_path()
and $VC_MIN_FREE_GB instead and do not need the shim.
"""
import os
import shutil

_ROOT = os.environ.get('VC_ROOT', os.path.join(os.path.expanduser('~'), 'vesuvius'))
_GUARD = os.environ.get('POD_GUARD_PATH', '').rstrip('/')

if _GUARD and os.path.exists(os.path.join(_ROOT, '.POD')):
    _orig_disk_usage = shutil.disk_usage

    def _pod_disk_usage(path):
        p = os.fspath(path)
        if p == _GUARD or p.startswith(_GUARD + '/'):
            u = _orig_disk_usage(_ROOT)
            off = int(float(os.environ.get('POD_GUARD_OFFSET_GB', '110')) * 1e9)
            return type(u)(u.total + off, u.used, u.free + off)
        return _orig_disk_usage(path)

    shutil.disk_usage = _pod_disk_usage

"""Distribute one executable without inheriting build-directory NTFS labels."""
from pathlib import Path
import hashlib
import zipfile


def package_release():
    root = Path(__file__).resolve().parent.parent
    source = root / 'dist' / 'TouchSign.exe'
    target = root / 'dist' / 'TouchSign-NERO-SingleFile.zip'
    payload = source.read_bytes()
    entry = zipfile.ZipInfo('TouchSign.exe')
    entry.create_system = 0  # MS-DOS metadata; no NTFS ACL or integrity label.
    entry.external_attr = 0x20
    entry.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr(entry, payload)
    with zipfile.ZipFile(target) as archive:
        assert archive.namelist() == ['TouchSign.exe']
        assert archive.testzip() is None
        assert not archive.getinfo('TouchSign.exe').extra
        assert hashlib.sha256(archive.read('TouchSign.exe')).digest() == hashlib.sha256(payload).digest()
    print(f'One-EXE release verified: {target} ({target.stat().st_size / 1024**2:.1f} MiB)')


if __name__ == '__main__':
    package_release()

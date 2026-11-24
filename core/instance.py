"""Linux desktop ownership. The kernel releases the lock on exit or crash."""
from contextlib import contextmanager
import json
import os


@contextmanager
def desktop_instance(path, url):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never unlink this file: another process may already be waiting on its inode.
    with path.open('a+', encoding='utf-8') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.seek(0)
            try:
                existing = json.load(handle)
            except (ValueError, OSError):
                existing = {}
            yield existing if isinstance(existing, dict) else {}
            return
        try:
            handle.seek(0)
            handle.truncate()
            json.dump({'pid': os.getpid(), 'url': url}, handle)
            handle.flush()
            yield None
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)

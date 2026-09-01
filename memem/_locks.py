"""_locks — cross-process flock shim: real fcntl locks on POSIX, no-op on Windows.

memem's locks guard concurrent writers (plugin hook + CLI + a host app like sliceagent).
``fcntl`` doesn't exist on Windows; per the Hermes pattern (file_sync.py: "Windows — file
locking skipped") the lock degrades to a no-op there — single-user risk accepted;
``msvcrt.locking`` is the upgrade path if it ever matters.
"""
try:
    import fcntl as _fcntl
except ImportError:  # Windows
    _fcntl = None  # type: ignore[assignment]


def lock_ex(fd) -> None:
    if _fcntl is not None:
        _fcntl.flock(fd, _fcntl.LOCK_EX)


def lock_ex_nb(fd) -> None:
    """Non-blocking exclusive lock; raises OSError if held, like flock(LOCK_NB).

    On Windows this always succeeds — callers read OSError as "another run
    holds the lock", and without real locks we cannot detect that. Same
    single-user tradeoff as the rest of this shim.
    """
    if _fcntl is not None:
        _fcntl.flock(fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)


def lock_sh(fd) -> None:
    if _fcntl is not None:
        _fcntl.flock(fd, _fcntl.LOCK_SH)


def unlock(fd) -> None:
    if _fcntl is not None:
        _fcntl.flock(fd, _fcntl.LOCK_UN)

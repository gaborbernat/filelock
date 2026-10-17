from __future__ import annotations

import gc
import itertools
import os
import subprocess  # ruff:ignore[suspicious-subprocess-import]  # only the fixed-argument, no-shell run below
import sys
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypeVar

import pytest
from capabilities import CAPABILITIES

from filelock import BaseFileLock, Timeout

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator

    from pytest_mock import MockerFixture

# Coverage restarts in every patched subprocess and re-imports the plugin there; pytest's pythonpath reaches only
# this process, so export it.
os.environ["PYTHONPATH"] = os.pathsep.join(
    part for part in (str(Path(__file__).parent.parent / "tasks"), os.environ.get("PYTHONPATH")) if part
)

_LockT = TypeVar("_LockT", bound=BaseFileLock)


def acquire_after_reclaim(lock: BaseFileLock) -> None:
    """
    Take a path whose stale marker the lock has to break first, without a wall-clock budget.

    A try finds the marker and breaks it, or first watches an unreadable one age; the last creates its own. A deadline
    would test how fast the machine is: a first try that outlasts it times out after the marker is gone and before the
    lock is held.
    """
    for _ in range(3):
        with suppress(Timeout):
            lock.acquire(blocking=False)
        if lock.is_locked:
            return


@contextmanager
def held_after_reclaim(lock: _LockT) -> Generator[_LockT]:
    """Hold *lock* through :func:`acquire_after_reclaim` and release it on exit."""
    acquire_after_reclaim(lock)
    try:
        yield lock
    finally:
        lock.release()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    # StrictSoftFileLock publishes claims with hard links, so tests marked requires_hard_links cannot run where
    # os.link is missing (Termux/Android CPython ships without it). The no-hard-link degradation itself is covered by
    # the os.link tests in test_filelock.py, which carry no marker and run everywhere.
    if not hasattr(os, "link"):  # pragma: no cover  # the body runs only on Termux/Android, absent from the CI matrix
        skip = pytest.mark.skip(reason="StrictSoftFileLock requires os.link (hard links), absent on Termux/Android")
        for item in items:
            if item.get_closest_marker("requires_hard_links") is not None:
                item.add_marker(skip)


@pytest.fixture(autouse=True, scope="session")
def _tracemalloc_lookup_without_support() -> Iterator[None]:
    # pytest's thread and unraisable exception hooks ask tracemalloc where the offending object was allocated. Where
    # that lookup raises (GraalPy), the hook reports "Failed to process thread exception" in place of the thread's real
    # error, so hand it the answer tracemalloc gives while it is not tracing.
    with pytest.MonkeyPatch.context() as patch:
        if not CAPABILITIES["tracemalloc-object-traceback"]:  # pragma: lacks tracemalloc-object-traceback
            patch.setattr("tracemalloc.get_object_traceback", lambda _source: None)
        yield


_HOST_COLLECTION_INTERVAL: Final[int] = 20
_HOST_COLLECTION_DUE: Final = itertools.cycle((False,) * (_HOST_COLLECTION_INTERVAL - 1) + (True,))


@pytest.fixture(autouse=True)
def _collect_garbage() -> Iterator[None]:
    # Multiprocessing objects register finalizers that close pipes, unlink semaphores and write to the resource
    # tracker; when cyclic garbage from one test is collected during a later one, those run against whatever that
    # later test has patched onto os.close or os.write. Collect after every test so finalizers fire in the test that
    # created the objects, before any such patch is in place. This runs after mocker's undo, since autouse fixtures
    # set up first and tear down last. Where a collection runs no finalizers (GraalPy queues them on the host
    # collector), it cannot do that, and a full host collection after every test was nearly all of the suite's runtime.
    # It still needs one now and then: with none at all, GraalPy CI runs stalled past the timeout inside trivial tests.
    yield
    if CAPABILITIES["collected-finalization"] or next(_HOST_COLLECTION_DUE):  # pragma: needs collected-finalization
        gc.collect()


@pytest.fixture
def close_failure(
    mocker: MockerFixture,
) -> Iterator[tuple[Callable[[int], None], OSError, RuntimeError]]:
    locked_fd: int | None = None
    release_error = OSError("release failed")
    release_cause = RuntimeError("release cause")
    release_error.__cause__ = release_cause
    release_error.__suppress_context__ = True
    real_close = os.close

    def capture(fd: int) -> None:
        nonlocal locked_fd
        locked_fd = fd
        mocker.patch("filelock._api.os.close", side_effect=close)

    def close(fd: int) -> None:
        # filelock._api.os is the os module, so this patches os.close process-wide. Raise only for the descriptor
        # under test. An unrelated close inside a GC finalizer would escape as an unraisable exception.
        if fd == locked_fd:
            raise release_error
        real_close(fd)  # pragma: no cover  # only an unrelated close (e.g. a GC finalizer) reaches here

    yield capture, release_error, release_cause
    if locked_fd is not None:  # pragma: no branch  # every consumer calls capture, so this is always set
        real_close(locked_fd)


@pytest.fixture
def inheriting_directory(tmp_path: Path) -> Path:  # pragma: win32 cover
    # A directory without inheritable entries gives a new file the token's default permissions, which icacls shows
    # like explicit ones; an inheritable entry for Users makes inherited entries show as (I).
    directory = tmp_path / "inheriting"
    directory.mkdir()
    subprocess.run(
        [Path(os.environ["SYSTEMROOT"], "System32", "icacls.exe"), directory, "/grant", "*S-1-5-32-545:(OI)(CI)(RX)"],
        capture_output=True,
        check=True,
    )
    return directory


@pytest.fixture
def hold_oplock() -> Iterator[Callable[[Path], Callable[[], None]]]:  # pragma: win32 cover
    # The marker skips the run off Windows; this only narrows ctypes.WinDLL for ty.
    assert sys.platform == "win32"
    import ctypes
    from ctypes import wintypes

    class Overlapped(ctypes.Structure):
        _fields_ = (
            ("Internal", ctypes.c_void_p),
            ("InternalHigh", ctypes.c_void_p),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        )

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = kernel32.CreateEventW.restype = wintypes.HANDLE
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.DeviceIoControl.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.POINTER(Overlapped),
    ]
    closers: list[Callable[[], None]] = []

    def hold(path: Path) -> Callable[[], None]:
        generic_read, share_all, open_existing, flag_overlapped = 0x80000000, 0x1 | 0x2 | 0x4, 3, 0x40000000
        handle = kernel32.CreateFileW(str(path), generic_read, share_all, None, open_existing, flag_overlapped, None)
        overlapped = Overlapped(hEvent=kernel32.CreateEventW(None, True, False, None))
        # FSCTL_REQUEST_OPLOCK_LEVEL_1 stays pending while granted; a read handle suffices to make other opens wait.
        assert not kernel32.DeviceIoControl(handle, 0x00090000, None, 0, None, 0, None, ctypes.byref(overlapped))
        assert ctypes.get_last_error() == 997  # ERROR_IO_PENDING
        closed = False

        def close() -> None:
            nonlocal closed
            if closed:
                return
            closed = True
            kernel32.CloseHandle(handle)
            # The request completes on close and writes overlapped, so wait before freeing it.
            kernel32.WaitForSingleObject(overlapped.hEvent, 5000)
            kernel32.CloseHandle(overlapped.hEvent)

        closers.append(close)
        return close

    yield hold
    for close in closers:
        close()


@pytest.fixture
def oplocked_lock_path(
    request: pytest.FixtureRequest, tmp_path: Path, hold_oplock: Callable[[Path], Callable[[], None]]
) -> Path:  # pragma: win32 cover
    lock_path = tmp_path / "a"
    lock_path.write_text(getattr(request, "param", ""), encoding="utf-8")
    hold_oplock(lock_path)
    return lock_path


@pytest.fixture
def working_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.chdir(first)
    return first, second

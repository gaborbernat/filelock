from __future__ import annotations

import multiprocessing as mp
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest

from filelock import Timeout
from filelock._read_write import ReadWriteLock

if TYPE_CHECKING:
    from pathlib import Path


def acquire_read_lock(
    lock_file: str,
    acquired_event: mp.Event,
    release_event: mp.Event | None = None,
    timeout: float = -1,
    blocking: bool = True,
    ready_event: mp.Event | None = None,
) -> bool:
    if ready_event:
        ready_event.wait(timeout=10)
    try:
        lock = ReadWriteLock(lock_file, timeout=timeout, blocking=blocking)
        with lock.read_lock():
            acquired_event.set()
            if release_event:
                release_event.wait(timeout=10)
            else:
                time.sleep(0.1)
        return True
    except Exception:
        return False


def acquire_write_lock(
    lock_file: str,
    acquired_event: mp.Event,
    release_event: mp.Event | None = None,
    timeout: float = -1,
    blocking: bool = True,
    ready_event: mp.Event | None = None,
) -> bool:
    if ready_event:
        ready_event.wait(timeout=10)
    try:
        lock = ReadWriteLock(lock_file, timeout=timeout, blocking=blocking)
        with lock.write_lock():
            acquired_event.set()
            if release_event:
                release_event.wait(timeout=10)
            else:
                time.sleep(0.1)
        return True
    except Exception:
        return False


def _try_upgrade_lock(
    lock_file: str,
    read_acquired_event: mp.Event,
    upgrade_attempted_event: mp.Event,
    upgrade_result: mp.Value,
) -> None:
    lock = ReadWriteLock(lock_file)
    try:
        with lock.read_lock():
            read_acquired_event.set()
            upgrade_attempted_event.set()
            try:
                with lock.write_lock(timeout=0.5):
                    upgrade_result.value = 1
            except RuntimeError:
                upgrade_result.value = 0
            except Timeout:
                upgrade_result.value = 2
            except Exception:
                upgrade_result.value = 3
    except Exception:
        upgrade_result.value = 4


def _try_downgrade_lock(
    lock_file: str,
    write_acquired_event: mp.Event,
    downgrade_attempted_event: mp.Event,
    downgrade_result: mp.Value,
) -> None:
    lock = ReadWriteLock(lock_file)
    try:
        with lock.write_lock():
            write_acquired_event.set()
            downgrade_attempted_event.set()
            try:
                with lock.read_lock(timeout=0.5):
                    downgrade_result.value = 1
            except RuntimeError:
                downgrade_result.value = 0
            except Timeout:
                downgrade_result.value = 2
            except Exception:
                downgrade_result.value = 3
    except Exception:
        downgrade_result.value = 4


def _recursive_read_lock(lock_file: str, success_flag: mp.Value) -> None:
    lock = ReadWriteLock(lock_file)
    try:
        with lock.read_lock():
            assert lock._lock_level == 1
            assert lock._current_mode == "read"
            with lock.read_lock():
                assert lock._lock_level == 2
                assert lock._current_mode == "read"
                with lock.read_lock():
                    assert lock._lock_level == 3
                    assert lock._current_mode == "read"
                assert lock._lock_level == 2
                assert lock._current_mode == "read"
            assert lock._lock_level == 1
            assert lock._current_mode == "read"
        assert lock._lock_level == 0
        assert lock._current_mode is None
        success_flag.value = 1
    except Exception:
        success_flag.value = 0


def _recursive_write_lock(lock_file: str, success_flag: mp.Value) -> None:
    lock = ReadWriteLock(lock_file)
    try:
        with lock.write_lock():
            assert lock._lock_level == 1
            assert lock._current_mode == "write"
            with lock.write_lock():
                assert lock._lock_level == 2
                assert lock._current_mode == "write"
                with lock.write_lock():
                    assert lock._lock_level == 3
                    assert lock._current_mode == "write"
                assert lock._lock_level == 2
                assert lock._current_mode == "write"
            assert lock._lock_level == 1
            assert lock._current_mode == "write"
        assert lock._lock_level == 0
        assert lock._current_mode is None
        success_flag.value = 1
    except Exception:
        success_flag.value = 0


def _acquire_lock_and_crash(lock_file: str, acquired_event: mp.Event, mode: str) -> None:
    lock = ReadWriteLock(lock_file)
    ctx = lock.write_lock() if mode == "write" else lock.read_lock()
    with ctx:
        acquired_event.set()
        while True:
            time.sleep(0.1)


def _chain_reader(
    idx: int,
    lock_file: str,
    release_count: mp.Value,
    forward_wait: mp.Event,
    backward_wait: mp.Event,
    forward_set: mp.Event | None,
    backward_set: mp.Event,
) -> None:
    forward_wait.wait(timeout=10)
    try:
        lock = ReadWriteLock(lock_file)
        with lock.read_lock():
            if idx > 0:
                time.sleep(0.5)

            if forward_set is not None:
                forward_set.set()

            if idx == 0:
                time.sleep(0.2)

            backward_set.set()
            backward_wait.wait(timeout=10)

            with release_count.get_lock():
                release_count.value += 1
    except Exception:
        pass


@contextmanager
def cleanup_processes(processes: list[mp.Process]):
    try:
        yield
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
            p.join(timeout=1)


@pytest.fixture
def lock_file(tmp_path: Path) -> str:
    return str(tmp_path / "test_lock.db")


@pytest.mark.timeout(10)
def test_read_locks_are_shared(lock_file: str) -> None:
    read1_acquired = mp.Event()
    read2_acquired = mp.Event()

    p1 = mp.Process(target=acquire_read_lock, args=(lock_file, read1_acquired))
    p2 = mp.Process(target=acquire_read_lock, args=(lock_file, read2_acquired))

    with cleanup_processes([p1, p2]):
        p1.start()
        p2.start()
        assert read1_acquired.wait(timeout=5), f"First read lock not acquired on {lock_file}"
        assert read2_acquired.wait(timeout=5), f"Second read lock not acquired on {lock_file}"
        p1.join(timeout=2)
        p2.join(timeout=2)
        assert not p1.is_alive()
        assert not p2.is_alive()


@pytest.mark.timeout(10)
def test_write_lock_excludes_other_write_locks(lock_file: str) -> None:
    write1_acquired = mp.Event()
    release_write1 = mp.Event()
    write2_acquired = mp.Event()

    p1 = mp.Process(target=acquire_write_lock, args=(lock_file, write1_acquired, release_write1))
    p2 = mp.Process(target=acquire_write_lock, args=(lock_file, write2_acquired, None, 0.5, True))

    with cleanup_processes([p1]):
        p1.start()
        assert write1_acquired.wait(timeout=5)

        with cleanup_processes([p2]):
            p2.start()
            assert not write2_acquired.wait(timeout=1)
            release_write1.set()
            p1.join(timeout=2)

        write2_acquired.clear()
        p3 = mp.Process(target=acquire_write_lock, args=(lock_file, write2_acquired, None))
        with cleanup_processes([p3]):
            p3.start()
            assert write2_acquired.wait(timeout=5)
            p3.join(timeout=2)
            assert not p3.is_alive()


@pytest.mark.timeout(10)
def test_write_lock_excludes_read_locks(lock_file: str) -> None:
    write_acquired = mp.Event()
    release_write = mp.Event()
    read_acquired = mp.Event()
    read_started = mp.Event()

    p1 = mp.Process(target=acquire_write_lock, args=(lock_file, write_acquired, release_write))
    p2 = mp.Process(target=acquire_read_lock, args=(lock_file, read_acquired, None, -1, True, read_started))

    with cleanup_processes([p1, p2]):
        p1.start()
        assert write_acquired.wait(timeout=5)

        p2.start()
        read_started.set()

        assert not read_acquired.wait(timeout=0.5)

        release_write.set()
        p1.join(timeout=2)

        assert read_acquired.wait(timeout=5)
        p2.join(timeout=2)
        assert not p2.is_alive()


@pytest.mark.timeout(10)
def test_read_lock_excludes_write_locks(lock_file: str) -> None:
    read_acquired = mp.Event()
    release_read = mp.Event()
    write_acquired = mp.Event()
    write_started = mp.Event()

    p1 = mp.Process(target=acquire_read_lock, args=(lock_file, read_acquired, release_read))
    p2 = mp.Process(target=acquire_write_lock, args=(lock_file, write_acquired, None, -1, True, write_started))

    with cleanup_processes([p1, p2]):
        p1.start()
        assert read_acquired.wait(timeout=5)

        p2.start()
        write_started.set()

        assert not write_acquired.wait(timeout=0.5)

        release_read.set()
        p1.join(timeout=2)

        assert write_acquired.wait(timeout=5)
        p2.join(timeout=2)
        assert not p2.is_alive()


@pytest.mark.timeout(30)
def test_write_non_starvation(lock_file: str) -> None:
    num_readers = 5

    chain_forward = [mp.Event() for _ in range(num_readers)]
    chain_backward = [mp.Event() for _ in range(num_readers)]
    writer_ready = mp.Event()
    writer_acquired = mp.Event()
    release_count = mp.Value("i", 0)

    readers: list[mp.Process] = []
    for i in range(num_readers):
        forward_set = chain_forward[i + 1] if i < num_readers - 1 else None
        backward_set = chain_backward[i - 1] if i > 0 else writer_ready
        reader = mp.Process(
            target=_chain_reader,
            args=(i, lock_file, release_count, chain_forward[i], chain_backward[i], forward_set, backward_set),
        )
        readers.append(reader)

    writer = mp.Process(target=acquire_write_lock, args=(lock_file, writer_acquired, None, 15, True, writer_ready))

    with cleanup_processes([*readers, writer]):
        for reader in readers:
            reader.start()

        chain_forward[0].set()
        assert writer_ready.wait(timeout=10)

        writer.start()
        assert writer_acquired.wait(timeout=15)

        with release_count.get_lock():
            read_releases = release_count.value
        assert read_releases < 3, f"Writer acquired after {read_releases} readers released - starvation"

        writer.join(timeout=2)
        assert not writer.is_alive()

        chain_backward[-1].set()
        for idx, reader in enumerate(readers):
            reader.join(timeout=3)
            assert not reader.is_alive(), f"Reader {idx} did not exit cleanly"


@pytest.mark.timeout(5)
def test_recursive_read_lock_acquisition(lock_file: str) -> None:
    success = mp.Value("i", 0)
    p = mp.Process(target=_recursive_read_lock, args=(lock_file, success))
    with cleanup_processes([p]):
        p.start()
        p.join(timeout=5)
    assert success.value == 1, "Recursive read lock acquisition failed"


@pytest.mark.timeout(5)
def test_lock_upgrade_prohibited(lock_file: str) -> None:
    read_acquired = mp.Event()
    upgrade_attempted = mp.Event()
    upgrade_result = mp.Value("i", -1)

    p = mp.Process(target=_try_upgrade_lock, args=(lock_file, read_acquired, upgrade_attempted, upgrade_result))
    with cleanup_processes([p]):
        p.start()
        assert read_acquired.wait(timeout=5)
        assert upgrade_attempted.wait(timeout=5)
        p.join(timeout=2)
        assert not p.is_alive()
    assert upgrade_result.value == 0, "Read lock was incorrectly upgraded to write lock"


@pytest.mark.timeout(5)
def test_lock_downgrade_prohibited(lock_file: str) -> None:
    write_acquired = mp.Event()
    downgrade_attempted = mp.Event()
    downgrade_result = mp.Value("i", -1)

    p = mp.Process(target=_try_downgrade_lock, args=(lock_file, write_acquired, downgrade_attempted, downgrade_result))
    with cleanup_processes([p]):
        p.start()
        assert write_acquired.wait(timeout=5)
        assert downgrade_attempted.wait(timeout=5)
        p.join(timeout=2)
        assert not p.is_alive()
    assert downgrade_result.value == 0, "Write lock was incorrectly downgraded to read lock"


@pytest.mark.timeout(10)
def test_timeout_behavior(lock_file: str) -> None:
    write_acquired = mp.Event()
    release_write = mp.Event()
    read_acquired = mp.Event()

    p1 = mp.Process(target=acquire_write_lock, args=(lock_file, write_acquired, release_write))
    p2 = mp.Process(target=acquire_read_lock, args=(lock_file, read_acquired, None, 0.5, True))

    with cleanup_processes([p1, p2]):
        p1.start()
        assert write_acquired.wait(timeout=5)

        start_time = time.time()
        p2.start()

        assert not read_acquired.wait(timeout=1)
        p2.join(timeout=5)

        elapsed = time.time() - start_time
        assert 0.4 <= elapsed <= 10.0, f"Timeout was not respected: {elapsed}s"

        release_write.set()
        p1.join(timeout=2)


@pytest.mark.timeout(10)
def test_non_blocking_behavior(lock_file: str) -> None:
    write_acquired = mp.Event()
    release_write = mp.Event()

    p1 = mp.Process(target=acquire_write_lock, args=(lock_file, write_acquired, release_write))

    with cleanup_processes([p1]):
        p1.start()
        assert write_acquired.wait(timeout=5)

        lock = ReadWriteLock(lock_file)

        start_time = time.time()
        with pytest.raises(Timeout), lock.read_lock(blocking=False):
            pytest.fail("Non-blocking read lock was unexpectedly acquired")

        elapsed = time.time() - start_time
        assert elapsed < 0.1, f"Non-blocking took too long: {elapsed}s"

        release_write.set()
        p1.join(timeout=2)


@pytest.mark.timeout(5)
def test_recursive_write_lock_acquisition(lock_file: str) -> None:
    success = mp.Value("i", 0)
    p = mp.Process(target=_recursive_write_lock, args=(lock_file, success))
    with cleanup_processes([p]):
        p.start()
        p.join(timeout=5)
    assert success.value == 1, "Recursive write lock acquisition failed"


@pytest.mark.timeout(10)
def test_write_lock_release_on_process_termination(lock_file: str) -> None:
    lock_acquired = mp.Event()

    p1 = mp.Process(target=_acquire_lock_and_crash, args=(lock_file, lock_acquired, "write"))
    p1.start()
    assert lock_acquired.wait(timeout=5)

    write_acquired = mp.Event()
    p2 = mp.Process(target=acquire_write_lock, args=(lock_file, write_acquired))

    with cleanup_processes([p1, p2]):
        time.sleep(0.2)
        p1.terminate()
        p1.join(timeout=2)

        p2.start()
        assert write_acquired.wait(timeout=5)
        p2.join(timeout=2)
        assert not p2.is_alive()


@pytest.mark.timeout(10)
def test_read_lock_release_on_process_termination(lock_file: str) -> None:
    lock_acquired = mp.Event()

    p1 = mp.Process(target=_acquire_lock_and_crash, args=(lock_file, lock_acquired, "read"))
    p1.start()
    assert lock_acquired.wait(timeout=5)

    write_acquired = mp.Event()
    p2 = mp.Process(target=acquire_write_lock, args=(lock_file, write_acquired))

    with cleanup_processes([p1, p2]):
        time.sleep(0.2)
        p1.terminate()
        p1.join(timeout=2)

        p2.start()
        assert write_acquired.wait(timeout=5)
        p2.join(timeout=2)
        assert not p2.is_alive()


@pytest.mark.timeout(5)
def test_single_read_lock_acquire_release(lock_file: str) -> None:
    lock = ReadWriteLock(lock_file)

    with lock.read_lock(), lock.read_lock():
        pass

    with lock.read_lock():
        pass


@pytest.mark.timeout(5)
def test_single_write_lock_acquire_release(lock_file: str) -> None:
    lock = ReadWriteLock(lock_file)

    with lock.write_lock(), lock.write_lock():
        pass

    with lock.write_lock():
        pass


@pytest.mark.timeout(5)
def test_read_then_write_lock(lock_file: str) -> None:
    lock = ReadWriteLock(lock_file)

    with lock.read_lock():
        pass

    with lock.write_lock():
        pass


@pytest.mark.timeout(5)
def test_write_then_read_lock(lock_file: str) -> None:
    lock = ReadWriteLock(lock_file)

    with lock.write_lock():
        pass

    with lock.read_lock():
        pass

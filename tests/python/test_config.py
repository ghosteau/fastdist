import pytest

from fastdist import config


class FakeNVMLError(Exception):
    pass


class _Mem:
    def __init__(self, free):
        self.free = free


def _install(monkeypatch, fake):
    monkeypatch.setattr(config, "pynvml", fake)
    monkeypatch.setattr(config, "_NVML_STATE", None)


def test_no_driver_returns_silently(monkeypatch):
    class FakeNVML:
        NVMLError = FakeNVMLError

        @staticmethod
        def nvmlInit():
            raise FakeNVMLError("library not found")

    _install(monkeypatch, FakeNVML)
    config.validate_gpu_capacity(1_000_000, 8)  # must not raise


def test_pynvml_not_installed_returns_silently(monkeypatch):
    _install(monkeypatch, None)
    config.validate_gpu_capacity(1_000_000, 8)  # must not raise


def test_raises_when_insufficient_memory(monkeypatch):
    class FakeNVML:
        NVMLError = FakeNVMLError

        @staticmethod
        def nvmlInit():
            return None

        @staticmethod
        def nvmlDeviceGetHandleByIndex(index):
            return object()

        @staticmethod
        def nvmlDeviceGetMemoryInfo(handle):
            return _Mem(free=1_000)

    _install(monkeypatch, FakeNVML)
    with pytest.raises(MemoryError, match="GPU Memory Overflow"):
        config.validate_gpu_capacity(1_000_000, 8)


def test_passes_when_memory_sufficient(monkeypatch):
    class FakeNVML:
        NVMLError = FakeNVMLError

        @staticmethod
        def nvmlInit():
            return None

        @staticmethod
        def nvmlDeviceGetHandleByIndex(index):
            return object()

        @staticmethod
        def nvmlDeviceGetMemoryInfo(handle):
            return _Mem(free=10 ** 12)

    _install(monkeypatch, FakeNVML)
    config.validate_gpu_capacity(1_000, 8)  # must not raise


def test_init_happens_only_once(monkeypatch):
    calls = {"n": 0}

    class FakeNVML:
        NVMLError = FakeNVMLError

        @staticmethod
        def nvmlInit():
            calls["n"] += 1

        @staticmethod
        def nvmlDeviceGetHandleByIndex(index):
            return object()

        @staticmethod
        def nvmlDeviceGetMemoryInfo(handle):
            return _Mem(free=10 ** 12)

    _install(monkeypatch, FakeNVML)
    for _ in range(5):
        config.validate_gpu_capacity(1_000, 8)
    assert calls["n"] == 1
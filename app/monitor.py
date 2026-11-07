from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty
from typing import Any

_GIB = 1024**3


@dataclass(frozen=True)
class _Sample:
    elapsed_seconds: float
    cpu_seconds: float
    memory_bytes: int


@dataclass(frozen=True)
class _GpuSample:
    elapsed_seconds: float
    device_count: int
    utilization_percent: float | None
    memory_used_bytes: int | None
    memory_total_bytes: int | None


@dataclass(frozen=True)
class _WorkloadToken:
    sample_index: int | None
    started_at: float


class _NvidiaSmiReader:
    def __init__(self) -> None:
        self.executable = shutil.which("nvidia-smi")
        self.unavailable_reason = (
            None if self.executable else "nvidia-smi is not available in the container"
        )

    def read(self) -> tuple[int, float | None, int | None, int | None]:
        if self.executable is None:
            raise RuntimeError(self.unavailable_reason or "nvidia-smi is unavailable")
        result = subprocess.run(
            [
                self.executable,
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        devices = list(csv.reader(io.StringIO(result.stdout)))
        if not devices:
            raise RuntimeError("nvidia-smi did not report any GPUs")

        utilizations: list[float] = []
        used_memory: list[int] = []
        total_memory: list[int] = []
        for device in devices:
            if len(device) < 3:
                continue
            utilization = _parse_optional_float(device[0])
            used_mib = _parse_optional_float(device[1])
            total_mib = _parse_optional_float(device[2])
            if utilization is not None:
                utilizations.append(utilization)
            if used_mib is not None:
                used_memory.append(round(used_mib * 1024**2))
            if total_mib is not None:
                total_memory.append(round(total_mib * 1024**2))
        if not devices:
            raise RuntimeError("nvidia-smi did not report any GPUs")
        return (
            len(devices),
            sum(utilizations) / len(utilizations) if utilizations else None,
            sum(used_memory) if len(used_memory) == len(devices) else None,
            sum(total_memory) if len(total_memory) == len(devices) else None,
        )


def _parse_optional_float(value: str) -> float | None:
    cleaned = value.strip()
    if not cleaned or cleaned.casefold() in {"n/a", "[n/a]", "not supported"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


class _WorkloadAccumulator:
    def __init__(self) -> None:
        self.documents = 0
        self.pages = 0
        self.duration_seconds = 0.0
        self.cpu_seconds = 0.0
        self.max_cpu_cores: float | None = None
        self.memory_byte_seconds = 0.0
        self.memory_duration_seconds = 0.0
        self.max_memory_bytes: int | None = None
        self.document_peak_bytes_total = 0
        self.gpu_samples: list[_GpuSample] = []

    def add(
        self,
        samples: list[_Sample],
        gpu_samples: list[_GpuSample],
        pages: int,
    ) -> None:
        self.documents += 1
        self.pages += max(0, pages)
        if samples:
            document_peak = max(sample.memory_bytes for sample in samples)
            self.document_peak_bytes_total += document_peak
            self.max_memory_bytes = max(self.max_memory_bytes or 0, document_peak)
        self.gpu_samples.extend(gpu_samples)

        for previous, current in zip(samples, samples[1:]):
            interval = current.elapsed_seconds - previous.elapsed_seconds
            if interval <= 0:
                continue
            cpu_delta = max(0.0, current.cpu_seconds - previous.cpu_seconds)
            self.duration_seconds += interval
            self.cpu_seconds += cpu_delta
            cores = cpu_delta / interval
            self.max_cpu_cores = max(self.max_cpu_cores or 0.0, cores)
            self.memory_byte_seconds += (
                (previous.memory_bytes + current.memory_bytes) / 2 * interval
            )
            self.memory_duration_seconds += interval

    def summary(self, *, capacity: float, baseline_memory_bytes: int | None) -> dict[str, Any]:
        average_cores = (
            self.cpu_seconds / self.duration_seconds if self.duration_seconds else None
        )
        average_memory = (
            round(self.memory_byte_seconds / self.memory_duration_seconds)
            if self.memory_duration_seconds
            else None
        )
        gpu_utilizations = [
            sample.utilization_percent
            for sample in self.gpu_samples
            if sample.utilization_percent is not None
        ]
        gpu_used = [
            sample.memory_used_bytes
            for sample in self.gpu_samples
            if sample.memory_used_bytes is not None
        ]
        gpu_total = [
            sample.memory_total_bytes
            for sample in self.gpu_samples
            if sample.memory_total_bytes is not None
        ]
        peak_memory = self.max_memory_bytes
        return {
            "documents": self.documents,
            "pages": self.pages,
            "measured_duration_seconds": round(self.duration_seconds, 3),
            "cpu": {
                "average_cores_used": round(average_cores, 3) if average_cores is not None else None,
                "max_cores_used": round(self.max_cpu_cores, 3) if self.max_cpu_cores is not None else None,
                "max_percent_of_capacity": (
                    round(self.max_cpu_cores / capacity * 100, 2)
                    if self.max_cpu_cores is not None and capacity > 0
                    else None
                ),
            },
            "memory": {
                "average_bytes": average_memory,
                "average_gib": round(average_memory / _GIB, 3) if average_memory is not None else None,
                "average_document_peak_bytes": (
                    round(self.document_peak_bytes_total / self.documents)
                    if self.documents and self.document_peak_bytes_total
                    else None
                ),
                "max_bytes": peak_memory,
                "max_gib": round(peak_memory / _GIB, 3) if peak_memory is not None else None,
                "max_incremental_bytes_above_baseline": (
                    max(0, peak_memory - baseline_memory_bytes)
                    if peak_memory is not None and baseline_memory_bytes is not None
                    else None
                ),
            },
            "gpu": {
                "sample_count": len(self.gpu_samples),
                "average_utilization_percent": (
                    round(sum(gpu_utilizations) / len(gpu_utilizations), 2)
                    if gpu_utilizations
                    else None
                ),
                "max_utilization_percent": max(gpu_utilizations) if gpu_utilizations else None,
                "max_memory_used_bytes": max(gpu_used) if gpu_used else None,
                "memory_total_bytes": max(gpu_total) if gpu_total else None,
            },
        }


class _ProcessPoolAccumulator:
    def __init__(self) -> None:
        self.sample_count = 0
        self.registered_workers: set[int] = set()
        self.cpu_seconds = 0.0
        self.cpu_duration_seconds = 0.0
        self.max_cpu_cores: float | None = None
        self.rss_byte_seconds = 0.0
        self.rss_duration_seconds = 0.0
        self.max_rss_bytes: int | None = None
        self.max_single_worker_rss_bytes: int | None = None
        self.previous: tuple[float, float, int] | None = None

    def register(self, pid: int) -> None:
        self.registered_workers.add(pid)

    def add(
        self,
        *,
        elapsed_seconds: float,
        cpu_seconds: float,
        rss_bytes: int,
        max_single_worker_rss_bytes: int,
    ) -> None:
        self.sample_count += 1
        self.max_rss_bytes = max(self.max_rss_bytes or 0, rss_bytes)
        self.max_single_worker_rss_bytes = max(
            self.max_single_worker_rss_bytes or 0,
            max_single_worker_rss_bytes,
        )
        if self.previous is not None:
            previous_elapsed, previous_cpu, previous_rss = self.previous
            interval = elapsed_seconds - previous_elapsed
            if interval > 0:
                cpu_delta = max(0.0, cpu_seconds - previous_cpu)
                self.cpu_seconds += cpu_delta
                self.cpu_duration_seconds += interval
                self.max_cpu_cores = max(
                    self.max_cpu_cores or 0.0,
                    cpu_delta / interval,
                )
                self.rss_byte_seconds += (previous_rss + rss_bytes) / 2 * interval
                self.rss_duration_seconds += interval
        self.previous = (elapsed_seconds, cpu_seconds, rss_bytes)

    def summary(self, *, capacity: float | None) -> dict[str, Any]:
        average_cores = (
            self.cpu_seconds / self.cpu_duration_seconds
            if self.cpu_duration_seconds
            else None
        )
        average_rss = (
            round(self.rss_byte_seconds / self.rss_duration_seconds)
            if self.rss_duration_seconds
            else None
        )
        return {
            "sample_count": self.sample_count,
            "worker_processes_registered": len(self.registered_workers),
            "cpu": {
                "average_cores_used": (
                    round(average_cores, 3) if average_cores is not None else None
                ),
                "max_cores_used": (
                    round(self.max_cpu_cores, 3)
                    if self.max_cpu_cores is not None
                    else None
                ),
                "max_percent_of_capacity": (
                    round(self.max_cpu_cores / capacity * 100, 2)
                    if self.max_cpu_cores is not None and capacity
                    else None
                ),
            },
            "memory": {
                "accounting": "sum of worker process-tree RSS; shared pages may be counted more than once",
                "average_process_tree_rss_bytes": average_rss,
                "average_process_tree_rss_gib": (
                    round(average_rss / _GIB, 3) if average_rss is not None else None
                ),
                "max_process_tree_rss_bytes": self.max_rss_bytes,
                "max_process_tree_rss_gib": (
                    round(self.max_rss_bytes / _GIB, 3)
                    if self.max_rss_bytes is not None
                    else None
                ),
                "max_single_worker_process_tree_rss_bytes": (
                    self.max_single_worker_rss_bytes
                ),
                "max_single_worker_process_tree_rss_gib": (
                    round(self.max_single_worker_rss_bytes / _GIB, 3)
                    if self.max_single_worker_rss_bytes is not None
                    else None
                ),
            },
        }


def _read_process_stat(pid: int) -> tuple[int, float] | None:
    try:
        raw_stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        fields = raw_stat[raw_stat.rfind(")") + 2 :].split()
        cpu_ticks = int(fields[11]) + int(fields[12])
        resident_pages = int(fields[21])
        clock_ticks = os.sysconf("SC_CLK_TCK")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return resident_pages * page_size, cpu_ticks / clock_ticks
    except (IndexError, OSError, ValueError):
        return None


def _process_tree_stats(root_pid: int) -> tuple[int, float] | None:
    pending = [root_pid]
    visited: set[int] = set()
    rss_bytes = 0
    cpu_seconds = 0.0
    found_process = False
    while pending:
        pid = pending.pop()
        if pid in visited:
            continue
        visited.add(pid)
        process_stat = _read_process_stat(pid)
        if process_stat is None:
            continue
        found_process = True
        process_rss, process_cpu = process_stat
        rss_bytes += process_rss
        cpu_seconds += process_cpu
        try:
            children_path = Path(f"/proc/{pid}/task/{pid}/children")
            pending.extend(
                int(child)
                for child in children_path.read_text(encoding="ascii").split()
            )
        except (OSError, ValueError):
            pass
    return (rss_bytes, cpu_seconds) if found_process else None


class _CgroupReader:
    def __init__(
        self,
        *,
        source: str,
        cpu_usage_path: Path,
        cpu_usage_scale: float,
        memory_usage_path: Path,
        cpu_capacity: float,
        memory_limit_bytes: int | None,
    ) -> None:
        self.source = source
        self.cpu_usage_path = cpu_usage_path
        self.cpu_usage_scale = cpu_usage_scale
        self.memory_usage_path = memory_usage_path
        self.cpu_capacity = cpu_capacity
        self.memory_limit_bytes = memory_limit_bytes

    def read(self) -> tuple[float, int]:
        cpu_seconds = self._read_cpu_seconds()
        memory_bytes = int(self.memory_usage_path.read_text(encoding="ascii").strip())
        return cpu_seconds, memory_bytes

    def _read_cpu_seconds(self) -> float:
        if self.source == "cgroup_v2":
            for line in self.cpu_usage_path.read_text(encoding="ascii").splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[0] == "usage_usec":
                    return int(parts[1]) / 1_000_000
            raise ValueError("cgroup v2 cpu.stat has no usage_usec value")
        return (
            int(self.cpu_usage_path.read_text(encoding="ascii").strip())
            * self.cpu_usage_scale
        )


def _available_cpu_count() -> float:
    try:
        affinity_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity_count = os.cpu_count() or 1
    return float(max(1, affinity_count))


def _cpu_capacity_v2(root: Path) -> float:
    capacity = _available_cpu_count()
    cpu_max = root / "cpu.max"
    if cpu_max.is_file():
        parts = cpu_max.read_text(encoding="ascii").split()
        if len(parts) == 2 and parts[0] != "max":
            quota, period = int(parts[0]), int(parts[1])
            if quota > 0 and period > 0:
                capacity = min(capacity, quota / period)
    return max(capacity, 0.01)


def _cpu_capacity_v1(cpu_dir: Path) -> float:
    capacity = _available_cpu_count()
    quota_path = cpu_dir / "cpu.cfs_quota_us"
    period_path = cpu_dir / "cpu.cfs_period_us"
    if quota_path.is_file() and period_path.is_file():
        quota = int(quota_path.read_text(encoding="ascii").strip())
        period = int(period_path.read_text(encoding="ascii").strip())
        if quota > 0 and period > 0:
            capacity = min(capacity, quota / period)
    return max(capacity, 0.01)


def _optional_limit(path: Path, *, unlimited_value: str = "max") -> int | None:
    if not path.is_file():
        return None
    value = path.read_text(encoding="ascii").strip()
    if value == unlimited_value:
        return None
    limit = int(value)
    if limit <= 0 or limit >= (1 << 60):
        return None
    return limit


def _find_cgroup_reader() -> _CgroupReader | None:
    root = Path("/sys/fs/cgroup")
    v2_cpu = root / "cpu.stat"
    v2_memory = root / "memory.current"
    if v2_cpu.is_file() and v2_memory.is_file():
        return _CgroupReader(
            source="cgroup_v2",
            cpu_usage_path=v2_cpu,
            cpu_usage_scale=1.0,
            memory_usage_path=v2_memory,
            cpu_capacity=_cpu_capacity_v2(root),
            memory_limit_bytes=_optional_limit(root / "memory.max"),
        )

    v1_layouts = (
        (root / "cpu,cpuacct", root / "memory"),
        (root / "cpuacct", root / "memory"),
    )
    for cpu_dir, memory_dir in v1_layouts:
        cpu_usage = cpu_dir / "cpuacct.usage"
        memory_usage = memory_dir / "memory.usage_in_bytes"
        if cpu_usage.is_file() and memory_usage.is_file():
            return _CgroupReader(
                source="cgroup_v1",
                cpu_usage_path=cpu_usage,
                cpu_usage_scale=1 / 1_000_000_000,
                memory_usage_path=memory_usage,
                cpu_capacity=_cpu_capacity_v1(cpu_dir),
                memory_limit_bytes=_optional_limit(
                    memory_dir / "memory.limit_in_bytes",
                    unlimited_value="-1",
                ),
            )
    return None


class ResourceMonitor:
    """Sample container resources and optional per-workload/pool usage."""

    def __init__(
        self,
        sample_interval_seconds: float = 1.0,
        gpu_sample_interval_seconds: float = 5.0,
    ) -> None:
        if sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be greater than zero")
        if gpu_sample_interval_seconds <= 0:
            raise ValueError("gpu_sample_interval_seconds must be greater than zero")
        self.sample_interval_seconds = sample_interval_seconds
        self.gpu_sample_interval_seconds = gpu_sample_interval_seconds
        try:
            self._reader = _find_cgroup_reader()
        except (OSError, ValueError):
            self._reader = None
        self._gpu_reader = _NvidiaSmiReader()
        self._gpu_unavailable_reason = self._gpu_reader.unavailable_reason
        self._samples: list[_Sample] = []
        self._gpu_samples: list[_GpuSample] = []
        self._samples_lock = threading.Lock()
        self._workload_lock = threading.Lock()
        self._workload_tracking_enabled = False
        self._process_pool_tracking_enabled = False
        self._process_pool_queue: Any | None = None
        self._process_pool_lock = threading.Lock()
        self._process_pools = {
            "native": _ProcessPoolAccumulator(),
            "ocr": _ProcessPoolAccumulator(),
        }
        self._workloads = {
            "native": _WorkloadAccumulator(),
            "ocr": _WorkloadAccumulator(),
            "unclassified": _WorkloadAccumulator(),
        }
        self._skipped_documents = 0
        self._baseline_memory_bytes: int | None = None
        self._last_gpu_sample_elapsed: float | None = None
        self._started_at: datetime | None = None
        self._started_monotonic: float | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def enable_workload_tracking(self, enabled: bool) -> None:
        if self._started_at is not None:
            raise RuntimeError("Workload tracking must be configured before starting")
        self._workload_tracking_enabled = enabled

    def enable_process_pool_tracking(self, registration_queue: Any) -> None:
        if self._started_at is not None:
            raise RuntimeError("Process-pool tracking must be configured before starting")
        self._process_pool_tracking_enabled = True
        self._process_pool_queue = registration_queue

    def start(self) -> None:
        if self._started_at is not None:
            raise RuntimeError("Resource monitor has already been started")
        self._started_at = datetime.now(timezone.utc)
        self._started_monotonic = time.monotonic()
        self._sample()
        self._thread = threading.Thread(
            target=self._sample_until_stopped,
            name="ocr-resource-monitor",
            daemon=True,
        )
        self._thread.start()

    def _sample_until_stopped(self) -> None:
        while not self._stop_event.wait(self.sample_interval_seconds):
            self._sample()

    def _sample(self) -> int | None:
        if self._started_monotonic is None:
            return None
        elapsed = time.monotonic() - self._started_monotonic
        sample_index: int | None = None
        if self._reader is not None:
            try:
                cpu_seconds, memory_bytes = self._reader.read()
            except (OSError, ValueError):
                pass
            else:
                with self._samples_lock:
                    self._samples.append(
                        _Sample(
                            elapsed_seconds=elapsed,
                            cpu_seconds=cpu_seconds,
                            memory_bytes=memory_bytes,
                        )
                    )
                    sample_index = len(self._samples) - 1
                    if self._baseline_memory_bytes is None:
                        self._baseline_memory_bytes = memory_bytes

        if (
            self._gpu_reader.executable
            and (
                self._last_gpu_sample_elapsed is None
                or elapsed - self._last_gpu_sample_elapsed >= self.gpu_sample_interval_seconds
            )
        ):
            self._last_gpu_sample_elapsed = elapsed
            try:
                device_count, utilization, used_memory, total_memory = self._gpu_reader.read()
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                self._gpu_unavailable_reason = str(error)
            else:
                self._gpu_unavailable_reason = None
                with self._samples_lock:
                    self._gpu_samples.append(
                        _GpuSample(
                            elapsed_seconds=elapsed,
                            device_count=device_count,
                            utilization_percent=utilization,
                            memory_used_bytes=used_memory,
                            memory_total_bytes=total_memory,
                        )
                    )
        self._sample_process_pools(elapsed)
        return sample_index

    def _sample_process_pools(self, elapsed: float) -> None:
        if not self._process_pool_tracking_enabled:
            return
        with self._process_pool_lock:
            if self._process_pool_queue is not None:
                while True:
                    try:
                        workload_type, pid = self._process_pool_queue.get_nowait()
                    except Empty:
                        break
                    except (EOFError, OSError, ValueError):
                        break
                    if workload_type in self._process_pools:
                        self._process_pools[workload_type].register(int(pid))

            for accumulator in self._process_pools.values():
                rss_bytes = 0
                cpu_seconds = 0.0
                worker_rss: list[int] = []
                for pid in accumulator.registered_workers:
                    stats = _process_tree_stats(pid)
                    if stats is None:
                        continue
                    worker_memory, worker_cpu = stats
                    rss_bytes += worker_memory
                    cpu_seconds += worker_cpu
                    worker_rss.append(worker_memory)
                if worker_rss:
                    accumulator.add(
                        elapsed_seconds=elapsed,
                        cpu_seconds=cpu_seconds,
                        rss_bytes=rss_bytes,
                        max_single_worker_rss_bytes=max(worker_rss),
                    )

    def begin_workload(self) -> _WorkloadToken:
        if not self._workload_tracking_enabled or self._started_monotonic is None:
            return _WorkloadToken(None, time.monotonic())
        sample_index = self._sample()
        return _WorkloadToken(
            sample_index,
            time.monotonic() - self._started_monotonic,
        )

    def finish_workload(
        self,
        token: _WorkloadToken,
        category: str,
        *,
        pages: int = 0,
        skipped: bool = False,
    ) -> None:
        if not self._workload_tracking_enabled:
            return
        end_index = self._sample()
        with self._samples_lock:
            if token.sample_index is not None and end_index is not None:
                samples = list(self._samples[token.sample_index : end_index + 1])
            else:
                samples = []
            end_elapsed = (
                samples[-1].elapsed_seconds
                if samples
                else time.monotonic() - (self._started_monotonic or time.monotonic())
            )
            gpu_samples = [
                sample
                for sample in self._gpu_samples
                if token.started_at <= sample.elapsed_seconds <= end_elapsed
            ]
        if skipped:
            with self._workload_lock:
                self._skipped_documents += 1
            return
        if category not in self._workloads:
            category = "unclassified"
        with self._workload_lock:
            self._workloads[category].add(samples, gpu_samples, pages)

    def stop(self) -> dict[str, Any]:
        if self._started_at is None or self._started_monotonic is None:
            raise RuntimeError("Resource monitor has not been started")
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join()
            self._thread = None
        self._sample()
        return self.snapshot(finished=True)

    def snapshot(self, *, finished: bool = False) -> dict[str, Any]:
        if self._started_at is None or self._started_monotonic is None:
            raise RuntimeError("Resource monitor has not been started")
        finished_at = datetime.now(timezone.utc)
        duration = max(0.0, time.monotonic() - self._started_monotonic)
        with self._samples_lock:
            samples = list(self._samples)
            gpu_samples = list(self._gpu_samples)
            baseline_memory_bytes = self._baseline_memory_bytes
        report = {
            "started_at": self._started_at.isoformat(),
            "updated_at": finished_at.isoformat(),
            "duration_seconds": round(duration, 3),
            "sample_interval_seconds": self.sample_interval_seconds,
            "sample_count": len(samples),
            "gpu_sample_interval_seconds": (
                self.gpu_sample_interval_seconds
                if self._gpu_reader.executable
                else None
            ),
            **self._resource_summary(samples, gpu_samples, baseline_memory_bytes),
            "process_pool_resources": self._process_pool_summary(),
        }
        if finished:
            report["finished_at"] = finished_at.isoformat()
        return report

    def _process_pool_summary(self) -> dict[str, Any] | None:
        if not self._process_pool_tracking_enabled:
            return None
        reader = self._reader
        capacity = reader.cpu_capacity if reader is not None else None
        with self._process_pool_lock:
            pools = {
                name: accumulator.summary(capacity=capacity)
                for name, accumulator in self._process_pools.items()
            }
        return {
            "resource_source": "worker process trees sampled from /proc",
            "memory_note": "RSS can count shared pages more than once and excludes container page cache; use container cgroup memory for the total",
            "pools": pools,
        }

    def _resource_summary(
        self,
        samples: list[_Sample],
        gpu_samples: list[_GpuSample],
        baseline_memory_bytes: int | None,
    ) -> dict[str, Any]:
        if self._reader is None or not samples:
            return {
                "resource_source": "unavailable",
                "cpu_capacity_cores": None,
                "memory_limit_bytes": None,
                "cpu": {
                    "average_cores_used": None,
                    "max_cores_used": None,
                    "average_percent_of_capacity": None,
                    "max_percent_of_capacity": None,
                },
                "memory": {
                    "average_bytes": None,
                    "average_gib": None,
                    "max_bytes": None,
                    "max_gib": None,
                    "baseline_bytes": baseline_memory_bytes,
                    "max_incremental_bytes_above_baseline": None,
                },
                "gpu": self._gpu_summary(gpu_samples),
                "workload_resources": self._workload_summary(
                    capacity=None,
                    baseline_memory_bytes=baseline_memory_bytes,
                ),
            }

        cpu_intervals: list[tuple[float, float]] = []
        weighted_memory = 0.0
        total_sampled_duration = 0.0
        for previous, current in zip(samples, samples[1:]):
            interval_duration = current.elapsed_seconds - previous.elapsed_seconds
            if interval_duration <= 0:
                continue
            cpu_delta = max(0.0, current.cpu_seconds - previous.cpu_seconds)
            cpu_intervals.append((cpu_delta / interval_duration, interval_duration))
            mean_interval_memory = (previous.memory_bytes + current.memory_bytes) / 2
            weighted_memory += mean_interval_memory * interval_duration
            total_sampled_duration += interval_duration

        capacity = self._reader.cpu_capacity
        if cpu_intervals:
            average_cores = sum(
                cores * interval for cores, interval in cpu_intervals
            ) / sum(interval for _, interval in cpu_intervals)
            max_cores = max(cores for cores, _ in cpu_intervals)
        else:
            average_cores = None
            max_cores = None

        memory_samples = [sample.memory_bytes for sample in samples]
        average_memory = (
            round(weighted_memory / total_sampled_duration)
            if total_sampled_duration > 0
            else round(sum(memory_samples) / len(memory_samples))
        )
        max_memory = max(memory_samples)
        return {
            "resource_source": self._reader.source,
            "cpu_capacity_cores": round(capacity, 3),
            "memory_limit_bytes": self._reader.memory_limit_bytes,
            "cpu": {
                "average_cores_used": (
                    round(average_cores, 3) if average_cores is not None else None
                ),
                "max_cores_used": round(max_cores, 3) if max_cores is not None else None,
                "average_percent_of_capacity": (
                    round(average_cores / capacity * 100, 2)
                    if average_cores is not None
                    else None
                ),
                "max_percent_of_capacity": (
                    round(max_cores / capacity * 100, 2)
                    if max_cores is not None
                    else None
                ),
            },
            "memory": {
                "average_bytes": average_memory,
                "average_gib": round(average_memory / _GIB, 3),
                "max_bytes": max_memory,
                "max_gib": round(max_memory / _GIB, 3),
                "baseline_bytes": baseline_memory_bytes,
                "max_incremental_bytes_above_baseline": (
                    max(0, max_memory - baseline_memory_bytes)
                    if baseline_memory_bytes is not None
                    else None
                ),
            },
            "gpu": self._gpu_summary(gpu_samples),
            "workload_resources": self._workload_summary(
                capacity=capacity,
                baseline_memory_bytes=baseline_memory_bytes,
            ),
        }

    def _gpu_summary(self, samples: list[_GpuSample]) -> dict[str, Any]:
        if not samples:
            return {
                "available": False,
                "source": "nvidia-smi",
                "scope": "device-wide; includes other host processes",
                "reason": self._gpu_unavailable_reason or "No GPU telemetry samples were collected",
                "device_count": None,
                "sample_count": 0,
                "average_utilization_percent": None,
                "max_utilization_percent": None,
                "max_memory_used_bytes": None,
                "memory_total_bytes": None,
            }
        utilizations = [
            sample.utilization_percent
            for sample in samples
            if sample.utilization_percent is not None
        ]
        used_memory = [
            sample.memory_used_bytes
            for sample in samples
            if sample.memory_used_bytes is not None
        ]
        total_memory = [
            sample.memory_total_bytes
            for sample in samples
            if sample.memory_total_bytes is not None
        ]
        return {
            "available": True,
            "source": "nvidia-smi",
            "scope": "device-wide; includes other host processes",
            "reason": None,
            "device_count": samples[-1].device_count,
            "sample_count": len(samples),
            "average_utilization_percent": (
                round(sum(utilizations) / len(utilizations), 2)
                if utilizations
                else None
            ),
            "max_utilization_percent": max(utilizations) if utilizations else None,
            "max_memory_used_bytes": max(used_memory) if used_memory else None,
            "memory_total_bytes": max(total_memory) if total_memory else None,
        }

    def _workload_summary(
        self,
        *,
        capacity: float | None,
        baseline_memory_bytes: int | None,
    ) -> dict[str, Any]:
        if not self._workload_tracking_enabled:
            return {
                "available": False,
                "reason": "Per-type attribution is measured only when OCR_WORKERS=1",
                "by_extraction_type": None,
            }
        with self._workload_lock:
            workload_stats = dict(self._workloads)
            skipped_documents = self._skipped_documents
            return {
                "available": True,
                "scope": "one serial worker; memory includes the shared process baseline",
                "classification": "OCR if any page used OCR; otherwise native extraction",
                "skipped_cached_documents": skipped_documents,
                "baseline_memory_bytes": baseline_memory_bytes,
                "by_extraction_type": {
                    name: stats.summary(
                        capacity=capacity or 0.0,
                        baseline_memory_bytes=baseline_memory_bytes,
                    )
                    for name, stats in workload_stats.items()
                },
            }

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Q4Kernels contributors.
"""CPU checks for benchmark selection and incremental result reporting."""

import csv
import io
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.ascend.mhc import bench_mhc_post, bench_mhc_pre


@pytest.fixture(params=[bench_mhc_pre, bench_mhc_post], ids=["pre", "post"])
def benchmark_module(request):
    return request.param


def test_default_selection_has_sixty_unique_shapes(benchmark_module, capsys):
    assert benchmark_module.main(["--list-cases"]) == 0
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    shapes = {(int(row["hidden"]), int(row["length"])) for row in rows}
    assert len(rows) == len(shapes) == 60
    assert {h for h, _ in shapes} == {4096, 5120, 7168}
    for hidden in (4096, 5120, 7168):
        lengths = sorted(n for h, n in shapes if h == hidden)
        assert len(lengths) == 20
        assert lengths[0] == 1 and lengths[-1] == 8192
        assert all(n <= 256 or n % 256 == 0 for n in lengths)


def test_case_listing_needs_no_torch_or_tilelang(benchmark_module):
    code = f"""
import sys
from {benchmark_module.__name__} import main
main(['--hidden-sizes', '4096', '--lengths', '32', '4096', '--list-cases'])
assert 'torch' not in sys.modules
assert 'torch_npu' not in sys.modules
assert 'tilelang' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[3],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.splitlines() == ["hidden,length", "4096,32", "4096,4096"]


def test_profile_duration_must_be_positive_and_finite(benchmark_module):
    assert benchmark_module._latency_us(SimpleNamespace(dur_ns=3500.0)) == 3.5
    for duration in (0, -1, float("nan"), float("inf")):
        with pytest.raises(RuntimeError, match="positive finite device duration"):
            benchmark_module._latency_us(SimpleNamespace(dur_ns=duration))
    # An event estimate or a profile without duration must not look like device timing.
    with pytest.raises(RuntimeError, match="positive finite device duration"):
        benchmark_module._latency_us(0.0035)


def test_csv_preserves_completed_cases_on_failure(benchmark_module, monkeypatch, tmp_path):
    calls = []

    def sample(n, hidden, args):
        calls.append((hidden, n, args.warmup, args.repeat))
        if n == 4096:
            raise RuntimeError("profiling failed")
        return 3.5

    monkeypatch.setattr(benchmark_module, "benchmark", sample)
    destination = tmp_path / "results" / "benchmark.csv"
    arguments = [
        "--hidden-sizes",
        "4096",
        "--lengths",
        "32",
        "4096",
        "--warmup",
        "5",
        "--repeat",
        "50",
        "--output",
        str(destination),
        "--log-file",
        str(tmp_path / "benchmark.log"),
    ]
    with pytest.raises(RuntimeError, match="H=4096, N=4096"):
        benchmark_module.main(arguments)
    assert calls == [(4096, 32, 5, 50), (4096, 4096, 5, 50)]
    saved = destination.read_text()
    assert list(csv.DictReader(io.StringIO(saved))) == [{"hidden": "4096", "length": "32", "device_mean_us": "3.5"}]
    assert "profiling failed" in (tmp_path / "benchmark.log").read_text()
    with pytest.raises(FileExistsError):
        benchmark_module.main(arguments)
    assert destination.read_text() == saved


def test_console_keeps_results_and_logs_python_native_and_child_output(benchmark_module, monkeypatch, tmp_path, capsys):
    def sample(n, hidden, args):
        print("Python compiler information")
        os.write(2, b"native compiler warning\n")
        subprocess.run([sys.executable, "-c", "print('child compiler warning')"], check=True)
        return 8.995

    monkeypatch.setattr(benchmark_module, "benchmark", sample)
    log = tmp_path / "benchmark.log"
    benchmark_module.main(["--hidden-sizes", "4096", "--lengths", "32", "--log-file", str(log)])
    console = capsys.readouterr()
    assert "H=4096 N=  32 ... device_mean_us=     8.995" in console.out
    assert "compiler" not in console.out + console.err
    saved = log.read_text()
    assert "Python compiler information" in saved
    assert "native compiler warning" in saved
    assert "child compiler warning" in saved


def test_log_capture_restores_native_streams_after_failure(tmp_path, capfd):
    from examples.ascend.mhc._bench import capture_output

    log_path = tmp_path / "failure.log"
    with log_path.open("w") as log, pytest.raises(ValueError, match="kernel failed"), capture_output(log):
        os.write(2, b"native error before failure\n")
        raise ValueError("kernel failed")
    os.write(2, b"console restored\n")
    assert capfd.readouterr().err == "console restored\n"
    assert "native error before failure" in log_path.read_text()
    assert "ValueError: kernel failed" in log_path.read_text()

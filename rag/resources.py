from __future__ import annotations

import os
import platform
import resource
import threading
import time


def configure_resources(settings):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(settings.cpu_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # A conservative hard address-space ceiling also bounds resident RAM.
    # It includes file mappings, so a process can fail earlier than an RSS-only cap.
    maximum = settings.ram_mb * 1024**2
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (min(maximum, hard) if hard >= 0 else maximum, hard))
    import psutil

    process = psutil.Process()
    stop = threading.Event()

    def monitor():
        while not stop.wait(0.1):
            try:
                rss = process.memory_info().rss + sum(p.memory_info().rss for p in process.children(recursive=True))
                if rss > settings.ram_mb * 1024**2:
                    os.write(2, b"RAM budget exceeded\n")
                    os._exit(137)
            except psutil.NoSuchProcess:
                return

    threading.Thread(target=monitor, daemon=True).start()
    return stop


def measurements(started):
    return {
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 3),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }


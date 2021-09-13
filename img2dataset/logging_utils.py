"""logging utils for the downloader"""

import wandb
import time
from collections import Counter


class CappedCounter:
    """Maintain a counter with a capping to avoid memory issues"""

    def __init__(self, max_size=10 ** 5):
        self.max_size = max_size
        self.counter = Counter()

    def increment(self, key):
        if len(self.counter) >= self.max_size:
            self._keep_most_frequent()
        self.counter[key] += 1

    def _keep_most_frequent(self):
        self.counter = Counter(dict(self.counter.most_common(int(self.max_size / 2))))

    def most_common(self, k):
        return self.counter.most_common(k)

    def update(self, counter):
        self.counter.update(counter.counter)
        if len(self.counter) >= self.max_size:
            self._keep_most_frequent()


class Logger:
    """logger which logs when number of calls reaches a value or a time interval has passed"""

    def __init__(self, processes_count=1, min_interval=0):
        """Log only every processes_count and if min_interval (seconds) have elapsed since last log"""
        # wait for all processes to return
        self.processes_count = processes_count
        self.processes_returned = 0
        # min time (in seconds) before logging a new table (avoids too many logs)
        self.min_interval = min_interval
        self.last = time.perf_counter()
        # keep track of whether we logged the last call
        self.last_call_logged = False
        self.last_args = None
        self.last_kwargs = None

    def __call__(self, *args, **kwargs):
        self.processes_returned += 1
        if self.processes_returned % self.processes_count == 0 and time.perf_counter() - self.last > self.min_interval:
            self.do_log(*args, **kwargs)
            self.last = time.perf_counter()
            self.last_call_logged = True
        else:
            self.last_call_logged = False
            self.last_args = args
            self.last_kwargs = kwargs

    def do_log(self, *args, **kwargs):
        raise NotImplementedError()

    def sync(self):
        """Ensure last call is logged"""
        if not self.last_call_logged:
            self.do_log(*self.last_args, **self.last_kwargs)
            # reset for next file
            self.processes_returned = 0


class SpeedLogger(Logger):
    """Log performance metrics"""

    def __init__(self, prefix, enable_wandb, **logger_args):
        super().__init__(**logger_args)
        self.prefix = prefix
        self.start = time.perf_counter()
        self.count = 0
        self.success = 0
        self.failed_to_download = 0
        self.failed_to_resize = 0
        self.enable_wandb = enable_wandb

    def __call__(
        self, duration, count, success, failed_to_download, failed_to_resize
    ):  # pylint: disable=arguments-differ
        self.count += count
        self.success += success
        self.failed_to_download += failed_to_download
        self.failed_to_resize += failed_to_resize
        super().__call__(duration, self.count, self.success, self.failed_to_download, self.failed_to_resize)

    def do_log(
        self, duration, count, success, failed_to_download, failed_to_resize
    ):  # pylint: disable=arguments-differ
        img_per_sec = count / duration
        success_ratio = 1.0 * success / count
        failed_to_download_ratio = 1.0 * failed_to_download / count
        failed_to_resize_ratio = 1.0 * failed_to_resize / count

        print(
            " - ".join(
                [
                    f"{self.prefix:<7}",
                    f"success: {success_ratio:.3f}",
                    f"failed to download: {failed_to_download_ratio:.3f}",
                    f"failed to resize: {failed_to_resize_ratio:.3f}",
                    f"images per sec: {img_per_sec:.0f}",
                    f"count: {count}",
                ]
            )
        )

        if self.enable_wandb:
            wandb.log(
                {
                    f"{self.prefix}/img_per_sec": img_per_sec,
                    f"{self.prefix}/success": success_ratio,
                    f"{self.prefix}/failed_to_download": failed_to_download_ratio,
                    f"{self.prefix}/failed_to_resize": failed_to_resize_ratio,
                    f"{self.prefix}/count": count,
                }
            )


class StatusTableLogger(Logger):
    """Log status table to W&B, up to `max_status` most frequent items"""

    def __init__(self, max_status=100, min_interval=60, enable_wandb=False, **logger_args):
        super().__init__(min_interval=min_interval, **logger_args)
        # avoids too many errors unique to a specific website (SSL certificates, etc)
        self.max_status = max_status
        self.enable_wandb = enable_wandb

    def do_log(self, status_dict, count):  # pylint: disable=arguments-differ
        if self.enable_wandb:
            status_table = wandb.Table(
                columns=["status", "frequency", "count"],
                data=[[k, 1.0 * v / count, v] for k, v in status_dict.most_common(self.max_status)],
            )
            wandb.run.log({"status": status_table})

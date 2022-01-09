"""logging utils for the downloader"""

import wandb
import time
from collections import Counter
import fsspec
import json
from multiprocessing import Process, Queue
import queue


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

    def dump(self):
        return self.counter

    @classmethod
    def load(cls, d, max_size=10 ** 5):
        c = CappedCounter(max_size)
        c.counter = Counter(d)
        return c


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


def write_stats(
    output_folder,
    shard_id,
    count,
    successes,
    failed_to_download,
    failed_to_resize,
    start_time,
    end_time,
    status_dict,
    oom_shard_count,
):
    """Write stats to disk"""
    stats = {
        "count": count,
        "successes": successes,
        "failed_to_download": failed_to_download,
        "failed_to_resize": failed_to_resize,
        "duration": end_time - start_time,
        "status_dict": status_dict.dump(),
    }
    fs, output_path = fsspec.core.url_to_fs(output_folder)
    shard_name = "{shard_id:0{oom_shard_count}d}".format(shard_id=shard_id, oom_shard_count=oom_shard_count)
    json_file = f"{output_path}/{shard_name}_stats.json"
    with fs.open(json_file, "w") as f:
        json.dump(stats, f, indent=4)


# https://docs.python.org/3/library/multiprocessing.html
# logger process that reads stats files regularly, aggregates and send to wandb / print to terminal
class LoggerProcess(Process):
    """Logger process that reads stats files regularly, aggregates and send to wandb / print to terminal"""

    def __init__(self, output_folder, enable_wandb, wandb_project, config_parameters, processes_count, log_interval=60):
        super().__init__()
        self.log_interval = log_interval
        self.enable_wandb = enable_wandb
        self.fs, self.output_path = fsspec.core.url_to_fs(output_folder)
        self.stats_files = set()
        self.wandb_project = wandb_project
        self.config_parameters = config_parameters
        self.processes_count = processes_count
        self.q = Queue()

    def run(self):
        """Run logger process"""

        if self.enable_wandb:
            self.current_run = wandb.init(project=self.wandb_project, config=self.config_parameters, anonymous="allow")
        else:
            self.current_run = None
        self.total_speed_logger = SpeedLogger(
            "total", processes_count=self.processes_count, enable_wandb=self.enable_wandb
        )
        self.status_table_logger = StatusTableLogger(
            processes_count=self.processes_count, enable_wandb=self.enable_wandb
        )
        start_time = time.perf_counter()
        last_check = 0
        total_status_dict = CappedCounter()
        while True:
            time.sleep(0.1)
            try:
                self.q.get(False)
                last_one = True
            except queue.Empty as _:
                last_one = False
            if not last_one and time.perf_counter() - last_check < self.log_interval:
                continue

            try:
                # read stats files
                stats_files = self.fs.glob(self.output_path + "/*.json")

                # get new stats files
                new_stats_files = set(stats_files) - self.stats_files
                if len(new_stats_files) == 0:
                    if last_one:
                        self.finish()
                        return

                # read new stats files
                for stats_file in new_stats_files:
                    with self.fs.open(stats_file, "r") as f:
                        stats = json.load(f)
                        SpeedLogger("worker", enable_wandb=self.enable_wandb)(
                            duration=stats["duration"],
                            count=stats["count"],
                            success=stats["successes"],
                            failed_to_download=stats["failed_to_download"],
                            failed_to_resize=stats["failed_to_resize"],
                        )
                        self.total_speed_logger(
                            duration=time.perf_counter() - start_time,
                            count=stats["count"],
                            success=stats["successes"],
                            failed_to_download=stats["failed_to_download"],
                            failed_to_resize=stats["failed_to_resize"],
                        )
                        status_dict = CappedCounter.load(stats["status_dict"])
                        total_status_dict.update(status_dict)
                        self.status_table_logger(total_status_dict, self.total_speed_logger.count)

                    self.stats_files.add(stats_file)
                last_check = time.perf_counter()

                if last_one:
                    self.finish()
                    return
            except Exception as e:  # pylint: disable=broad-except
                print(e)
                self.finish()
                return

    def finish(self):
        """Finish logger process"""
        self.total_speed_logger.sync()
        self.status_table_logger.sync()
        if self.current_run is not None:
            self.current_run.finish()

    def join(self, timeout=None):
        """Stop logger process"""
        self.q.put("stop")
        super().join()
        self.q.close()

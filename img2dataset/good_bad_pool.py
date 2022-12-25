import queue
from threading import Thread
import time

class GoodBadPool:
    def __init__(self, generator, runner, timeout, pool_size, out_queue_max) -> None:
        self.generator = generator
        self.runner = runner
        self.timeout = timeout
        self.pool_size = pool_size
        self.out_queue_max = out_queue_max
        self.results = []
        self.good_threads = []
        self.bad_threads = []
        self.outqueue = queue.SimpleQueue()
        self.good_done = {}
        self.item_left = True

    def call(self, start_time, item):
        result = self.runner(item)
        key = item[0]
        if time.time() - start_time < self.timeout:
            self.outqueue.put(result)
            self.good_done[key] = True

    def cleanup_bad_threads(self):
        still_bad_threads = []
        for thread in self.bad_threads:
            thread.join(0)
            if thread.is_alive():
                still_bad_threads.append(thread)
        self.bad_threads = still_bad_threads

    def cleanup_good_threads(self):
        # move slow threads to bad threads
        still_good_threads = []
        for start_time, key, thread in self.good_threads:
            if key in self.good_done:
                thread.join(0)
                del self.good_done[key]
                continue
            if time.time() - start_time > self.timeout:
                self.outqueue.put((key, None, "timeout"))
                self.bad_threads.append(thread)
            else:
                still_good_threads.append((start_time, key, thread))
        self.good_threads = still_good_threads


    def provider(self):
        """Loops infinitely, if we need new values, try to get them
        1. clean up bad threads (join them)
        2. clean up good threads by moving the slow ones to bad threads
        3. start new threads if possible
        """
        while True:
            if self.outqueue.qsize() > self.out_queue_max:
                time.sleep(0.1)
                continue

            self.cleanup_bad_threads()
            self.cleanup_good_threads()
            #print(f"good: {len(self.good_threads)}, bad: {len(self.bad_threads)}, outqueue: {self.outqueue.qsize()}")

            if self.item_left and len(self.good_threads) < self.pool_size:
                try:
                    item = next(self.generator)
                    key = item[0]
                except StopIteration:
                    self.item_left = False
                    continue
                start_time = time.time()
                thread = Thread(target=self.call, args=(start_time, item,))
                thread.start()
                self.good_threads.append((start_time, key, thread))
            else:
                if len(self.good_threads) == 0 and not self.item_left:
                    self.outqueue.put(None)
                    return
                time.sleep(0.1)
                continue

    def run(self):
        t = Thread(target=self.provider)
        t.start()
        while True:
            item = self.outqueue.get()
            if item is None:
                break
            yield item
        t.join(0)


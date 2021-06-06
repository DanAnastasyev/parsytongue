# -*- coding: utf-8 -*-

import attr
import time

from contextlib import contextmanager
from collections import defaultdict


@attr.s
class _PerfCounter(object):
    total_time = attr.ib(default=0.)
    total_accesses = attr.ib(default=0)
    subcounters = attr.ib()

    @subcounters.default
    def _default_subcounters(self):
        return defaultdict(_PerfCounter)

    def update(self, time):
        self.total_time += time
        self.total_accesses += 1


_PERF_COUNTERS = defaultdict(_PerfCounter)


@contextmanager
def timed(name):
    global _PERF_COUNTERS

    old_perf_counters = _PERF_COUNTERS
    _PERF_COUNTERS = old_perf_counters[name].subcounters

    start_time = time.time()
    try:
        yield
    finally:
        old_perf_counters[name].update(time.time() - start_time)
        _PERF_COUNTERS = old_perf_counters


def _show_perf_results(perf_counters, indent):
    sum_time = sum(perf_counter.total_time for perf_counter in perf_counters.values())
    for perf_counter_name, perf_counter in sorted(perf_counters.items(), key=lambda pair: pair[1].total_time, reverse=True):
        print('{}Time spent on {} is {:.2f}s ({:.2%}), which is {:.2f} it/s.'.format(
            ' ' * indent,
            perf_counter_name, perf_counter.total_time,
            perf_counter.total_time / sum_time,
            perf_counter.total_accesses / perf_counter.total_time
        ))
        _show_perf_results(perf_counter.subcounters, indent=indent + 4)


def show_perf_results():
    _show_perf_results(_PERF_COUNTERS, indent=0)


def clear_perf_counters():
    _PERF_COUNTERS.clear()

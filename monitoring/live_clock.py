"""Separate sampling duration, UTC adjustments and suspend/process gaps.

UTC is metadata, not the experiment's interval clock. A small UTC correction
must not be relabelled as proof that the laptop slept.
"""

import math
import time

from monitoring.psi import require


def boot_time():
    if hasattr(time, 'CLOCK_BOOTTIME'):
        try:
            return time.clock_gettime(time.CLOCK_BOOTTIME)
        except OSError:
            pass
    return None


class LiveClockGuard:
    VERSION = 2

    def __init__(self, start_mono, start_unix, start_boot=None):
        self.start_mono, self.start_unix, self.start_boot = start_mono, start_unix, start_boot
        self.previous_mono, self.previous_offset = start_mono, 0.0
        self.last_noticed_offset = 0.0
        self.utc_notices = 0
        self.latest_offset = 0.0
        self.maximum_abs_offset = 0.0
        self.maximum_loop_gap = 0.0
        self.maximum_suspend_gap = 0.0
        self.review_required = False
        self.last_interruption = None

    def _interrupt(self, kind, message, **details):
        self.last_interruption = {'kind': kind, **details}
        raise ValueError(message)

    def check(self, now, unix, boot=None):
        require(all(math.isfinite(v) for v in (now, unix)), 'Non-finite clock reading')
        gap = now - self.previous_mono
        if gap < 0:
            self._interrupt('backward_monotonic_clock', 'Sampling monotonic clock moved backwards', gap_seconds=gap)
        self.maximum_loop_gap = max(self.maximum_loop_gap, gap)
        suspend_gap = 0.0
        if self.start_boot is not None and boot is not None:
            require(math.isfinite(boot), 'Non-finite suspend-clock reading')
            suspend_gap = (boot - self.start_boot) - (now - self.start_mono)
            self.maximum_suspend_gap = max(self.maximum_suspend_gap, suspend_gap)
            if suspend_gap > 5:
                self._interrupt('suspend_clock_gap',
                                'Suspend-aware clock advanced over five seconds beyond the sampling clock; the live run was interrupted',
                                suspend_clock_gap_seconds=suspend_gap)
        if gap > 10:
            self._interrupt('process_gap', 'Live processing paused for over ten seconds; restart the measured experiment',
                            loop_gap_seconds=gap)
        offset = (unix - self.start_unix) - (now - self.start_mono)
        offset_step = offset - self.previous_offset
        self.latest_offset = offset
        self.maximum_abs_offset = max(self.maximum_abs_offset, abs(offset))
        # A VM/host pause can occasionally be invisible to both guest elapsed
        # clocks. A very large abrupt UTC jump is ambiguous: preserve data,
        # flag review, and never silently declare the run passing.
        if abs(offset_step) >= 60:
            self.review_required = True
        event = None
        if abs(offset - self.last_noticed_offset) >= 2 or abs(offset_step) >= 60:
            self.utc_notices += 1
            self.last_noticed_offset = offset
            event = {'kind': 'utc_clock_adjustment', 'guard_version': self.VERSION,
                     'elapsed_s': now - self.start_mono, 'observed_unix': unix,
                     'utc_minus_elapsed_seconds': offset, 'utc_offset_step_seconds': offset_step,
                     'clock_review_required': self.review_required,
                     'interval_clock': 'time.monotonic', 'suspend_clock_gap_seconds': suspend_gap}
        self.previous_mono, self.previous_offset = now, offset
        return event

    def summary(self):
        return {'guard_version': self.VERSION, 'interval_clock': 'time.monotonic',
                'suspend_clock': 'CLOCK_BOOTTIME' if self.start_boot is not None else 'unavailable',
                'utc_adjustment_notices': self.utc_notices,
                'latest_utc_minus_elapsed_seconds': self.latest_offset,
                'maximum_abs_utc_minus_elapsed_seconds': self.maximum_abs_offset,
                'maximum_processing_loop_gap_seconds': self.maximum_loop_gap,
                'maximum_suspend_clock_gap_seconds': self.maximum_suspend_gap,
                'clock_review_required': self.review_required,
                'last_interruption': self.last_interruption}

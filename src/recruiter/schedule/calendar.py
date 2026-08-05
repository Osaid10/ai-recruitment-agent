"""Stage 3a — calendar availability and booking.

`CalendarAdapter` is the seam. `MockCalendar` implements it with a deterministic
in-memory calendar so the whole pipeline demos without OAuth. A real
`GoogleCalendarAdapter` implementing the same three methods drops in without any
other file changing.

All datetimes here are naive local time. That keeps the demo free of a tzdata
dependency on Windows, and the .ics writer emits floating local time to match.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Protocol

from ..models import Interviewer, TimeSlot

WORKDAY_START = 9  # 09:00
WORKDAY_END = 17  # 17:00
LUNCH_START = 13
LUNCH_END = 14
SLOT_GRANULARITY_MINUTES = 30


class NoSlotAvailable(RuntimeError):
    """No time in the window works for the whole panel. Escalates to a human."""


class CalendarAdapter(Protocol):
    """Implement these three and the scheduler works against any calendar."""

    def busy(self, email: str, start: datetime, end: datetime) -> list[TimeSlot]:
        """Existing commitments for one person in the window."""

    def book(self, slot: TimeSlot, attendees: list[str], subject: str, body: str) -> str:
        """Create the event. Returns a provider event id."""

    def cancel(self, event_id: str) -> bool:
        """Remove a booking. Returns whether anything was removed."""


@dataclass
class MockCalendar:
    """An in-memory calendar with plausible, reproducible busy time.

    Busy blocks are derived from a hash of each attendee's email, so the same
    person always has the same schedule across runs — demos are repeatable, and
    the "no slot available" path can be triggered on purpose via `block_out`.
    """

    events: dict[str, tuple[TimeSlot, list[str], str]] = field(default_factory=dict)
    _manual_busy: dict[str, list[TimeSlot]] = field(default_factory=dict)
    seed: int = 7
    _counter: int = 0

    # -- adapter -----------------------------------------------------------

    def busy(self, email: str, start: datetime, end: datetime) -> list[TimeSlot]:
        blocks = [*self._synthetic_busy(email, start, end), *self._manual_busy.get(email, [])]
        for slot, attendees, _ in self.events.values():
            if email in attendees:
                blocks.append(slot)
        window = TimeSlot(start=start, end=end)
        return sorted(
            (b for b in blocks if b.overlaps(window)), key=lambda s: s.start
        )

    def book(self, slot: TimeSlot, attendees: list[str], subject: str, body: str) -> str:
        self._counter += 1
        event_id = f"mockcal-{self._counter:04d}"
        self.events[event_id] = (slot, attendees, subject)
        return event_id

    def cancel(self, event_id: str) -> bool:
        return self.events.pop(event_id, None) is not None

    # -- test helpers ------------------------------------------------------

    def block_out(self, email: str, slot: TimeSlot) -> None:
        """Mark someone busy. Used to exercise the no-availability path."""
        self._manual_busy.setdefault(email, []).append(slot)

    def clear(self) -> None:
        self.events.clear()
        self._manual_busy.clear()

    # -- internals ---------------------------------------------------------

    def _synthetic_busy(self, email: str, start: datetime, end: datetime) -> list[TimeSlot]:
        """Two recurring meetings per person, stable across runs."""
        digest = hashlib.md5(f"{email}:{self.seed}".encode()).digest()
        blocks: list[TimeSlot] = []
        day = start.replace(hour=0, minute=0, second=0, microsecond=0)

        while day <= end:
            if day.weekday() < 5:
                for i in range(2):
                    hour = WORKDAY_START + (digest[(day.day + i) % len(digest)] % 7)
                    if hour >= WORKDAY_END:
                        continue
                    block_start = day.replace(hour=hour, minute=0)
                    blocks.append(TimeSlot(start=block_start, end=block_start + timedelta(hours=1)))
            day += timedelta(days=1)
        return blocks


def _candidate_slots(
    window_start: datetime, window_end: datetime, duration: timedelta
) -> list[TimeSlot]:
    """Every working-hours slot in the window, on a 30-minute grid."""
    slots: list[TimeSlot] = []
    cursor = window_start.replace(minute=0, second=0, microsecond=0)
    if cursor < window_start:
        cursor += timedelta(hours=1)

    while cursor + duration <= window_end:
        end = cursor + duration
        day_start = cursor.replace(hour=WORKDAY_START, minute=0)
        day_end = cursor.replace(hour=WORKDAY_END, minute=0)
        lunch = TimeSlot(
            start=cursor.replace(hour=LUNCH_START, minute=0),
            end=cursor.replace(hour=LUNCH_END, minute=0),
        )
        slot = TimeSlot(start=cursor, end=end)

        within_workday = cursor.weekday() < 5 and cursor >= day_start and end <= day_end
        if within_workday and not slot.overlaps(lunch):
            slots.append(slot)
        cursor += timedelta(minutes=SLOT_GRANULARITY_MINUTES)
    return slots


def find_slot(
    adapter: CalendarAdapter,
    interviewers: list[Interviewer],
    window_start: datetime,
    window_end: datetime,
    duration_minutes: int = 45,
) -> TimeSlot:
    """First slot in the window where every interviewer is free.

    Raises `NoSlotAvailable` rather than guessing — an interview booked over
    someone's existing meeting is worse than asking a human to widen the window.
    """
    if not interviewers:
        raise NoSlotAvailable("no interviewers assigned to this role")
    if window_end <= window_start:
        raise NoSlotAvailable("scheduling window ends before it starts")

    busy_by_person = {
        person.email: adapter.busy(person.email, window_start, window_end)
        for person in interviewers
    }

    duration = timedelta(minutes=duration_minutes)
    for slot in _candidate_slots(window_start, window_end, duration):
        if all(
            not any(slot.overlaps(b) for b in blocks) for blocks in busy_by_person.values()
        ):
            return slot

    raise NoSlotAvailable(
        f"no {duration_minutes}-minute slot works for all {len(interviewers)} interviewers "
        f"between {window_start:%d %b %H:%M} and {window_end:%d %b %H:%M}"
    )


def default_window(days_ahead: int = 10, start_in_days: int = 1) -> tuple[datetime, datetime]:
    """A sensible booking window: from tomorrow, ten days out."""
    now = datetime.now()
    start = (now + timedelta(days=start_in_days)).replace(
        hour=WORKDAY_START, minute=0, second=0, microsecond=0
    )
    return start, start + timedelta(days=days_ahead)

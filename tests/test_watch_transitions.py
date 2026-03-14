from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import unittest

from trade_scout.watch_state import WatchLifecycleState, assert_valid_watch_lifecycle
from trade_scout.watch_transitions import WatchBar, WatchTransitionType, decide_watch_bar_transition


def make_watch(**overrides):
    base = dict(
        alert_id="a1",
        symbol="BTC/USD",
        timeframe_min=5,
        tier="A",
        created_ts=datetime(2026, 1, 1, tzinfo=timezone.utc),
        entry_low=100.0,
        entry_high=101.0,
        stop=95.0,
        t1=105.0,
        t2=110.0,
        R=6.0,
        score=7.5,
        max_favorable=101.0,
        max_adverse=101.0,
        hit_t1=False,
        hit_t2=False,
        done=False,
        done_reason=None,
        done_ts=None,
        entered=False,
        entry_ts=None,
        entry_px=None,
        t1_ts=None,
        t2_ts=None,
        stop_ts=None,
    )
    base.update(overrides)
    watch = SimpleNamespace(**base)
    assert_valid_watch_lifecycle(watch)
    return watch


class WatchTransitionsTest(unittest.TestCase):
    def test_pending_stays_pending(self):
        watch = make_watch()
        bar = WatchBar(high=99.0, low=98.0, close=98.5, ts=watch.created_ts + timedelta(minutes=1))

        decision = decide_watch_bar_transition(watch, bar, timedelta(hours=4))

        self.assertEqual(decision.state_before, WatchLifecycleState.PENDING_ENTRY)
        self.assertEqual(decision.state_after, WatchLifecycleState.PENDING_ENTRY)
        self.assertEqual(decision.transitions, ())
        self.assertTrue(decision.keep_active)

    def test_pending_enters(self):
        watch = make_watch()
        bar = WatchBar(high=102.0, low=100.5, close=101.5, ts=watch.created_ts + timedelta(minutes=1))

        decision = decide_watch_bar_transition(watch, bar, timedelta(hours=4))

        self.assertEqual([t.transition_type for t in decision.transitions], [WatchTransitionType.ENTER])
        self.assertEqual(decision.state_after, WatchLifecycleState.ENTERED)
        self.assertTrue(decision.keep_active)

    def test_pending_expires(self):
        watch = make_watch()
        bar = WatchBar(high=99.0, low=98.0, close=98.5, ts=watch.created_ts + timedelta(hours=4))

        decision = decide_watch_bar_transition(watch, bar, timedelta(hours=4))

        self.assertEqual([t.transition_type for t in decision.transitions], [WatchTransitionType.EXPIRE_PENDING])
        self.assertEqual(decision.state_after, WatchLifecycleState.EXPIRED)
        self.assertFalse(decision.keep_active)

    def test_entered_stays_entered(self):
        watch = make_watch(
            entered=True,
            entry_ts=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
            entry_px=101.0,
        )
        bar = WatchBar(high=104.0, low=100.0, close=103.0, ts=watch.created_ts + timedelta(minutes=2))

        decision = decide_watch_bar_transition(watch, bar, timedelta(hours=4))

        self.assertEqual(decision.state_before, WatchLifecycleState.ENTERED)
        self.assertEqual(decision.state_after, WatchLifecycleState.ENTERED)
        self.assertEqual(decision.transitions, ())
        self.assertTrue(decision.keep_active)

    def test_entered_exits_by_current_priority(self):
        watch = make_watch(
            entered=True,
            entry_ts=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
            entry_px=101.0,
        )
        bar = WatchBar(high=111.0, low=94.0, close=109.0, ts=watch.created_ts + timedelta(minutes=2))

        decision = decide_watch_bar_transition(watch, bar, timedelta(hours=4))

        self.assertEqual([t.transition_type for t in decision.transitions], [WatchTransitionType.EXIT_STOP])
        self.assertEqual(decision.state_after, WatchLifecycleState.EXITED)
        self.assertFalse(decision.keep_active)

    def test_terminal_states_are_terminal(self):
        exited = make_watch(
            entered=True,
            entry_ts=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
            entry_px=101.0,
            done=True,
            done_reason="STOP",
            done_ts=datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc),
            stop_ts=datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc),
        )
        expired = make_watch(
            done=True,
            done_reason="NO_ENTRY_FILL",
            done_ts=datetime(2026, 1, 1, 4, 0, tzinfo=timezone.utc),
        )
        bar = WatchBar(high=120.0, low=90.0, close=100.0, ts=datetime(2026, 1, 1, 5, 0, tzinfo=timezone.utc))

        exited_decision = decide_watch_bar_transition(exited, bar, timedelta(hours=4))
        expired_decision = decide_watch_bar_transition(expired, bar, timedelta(hours=4))

        self.assertEqual(exited_decision.state_before, WatchLifecycleState.EXITED)
        self.assertEqual(exited_decision.state_after, WatchLifecycleState.EXITED)
        self.assertEqual(exited_decision.transitions, ())
        self.assertFalse(exited_decision.keep_active)

        self.assertEqual(expired_decision.state_before, WatchLifecycleState.EXPIRED)
        self.assertEqual(expired_decision.state_after, WatchLifecycleState.EXPIRED)
        self.assertEqual(expired_decision.transitions, ())
        self.assertFalse(expired_decision.keep_active)


if __name__ == "__main__":
    unittest.main()

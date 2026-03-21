from datetime import datetime, timedelta, timezone
import unittest

from trade_scout.cooldown import Cooldown


class CooldownTest(unittest.TestCase):
    def test_cooldown_is_strategy_aware(self):
        cooldown = Cooldown(minutes=60)
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)

        cooldown.mark("BTC/USD", "A_CONTINUATION", now, "strategy_a")

        self.assertFalse(cooldown.allowed("BTC/USD", "A_CONTINUATION", now + timedelta(minutes=5), "strategy_a"))
        self.assertTrue(cooldown.allowed("BTC/USD", "A_CONTINUATION", now + timedelta(minutes=5), "strategy_b"))


if __name__ == "__main__":
    unittest.main()

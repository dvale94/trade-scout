from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

if "dotenv" not in sys.modules:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_stub

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")
    requests_stub.Response = object
    requests_stub.get = lambda *args, **kwargs: None
    requests_stub.post = lambda *args, **kwargs: None
    requests_stub.delete = lambda *args, **kwargs: None
    sys.modules["requests"] = requests_stub

from trade_scout.execution.broker import derive_crypto_stop_limit_price, submit_stop_order
from trade_scout.execution.models import BrokerOrderResult


class ExecutionBrokerTest(unittest.TestCase):
    def test_derive_crypto_stop_limit_price_offsets_long_and_short(self) -> None:
        self.assertEqual(100.1, derive_crypto_stop_limit_price(stop_price=100.0, side="LONG"))
        self.assertEqual(99.9, derive_crypto_stop_limit_price(stop_price=100.0, side="SHORT"))

    def test_submit_stop_order_uses_stop_limit_for_crypto(self) -> None:
        with patch(
            "trade_scout.execution.broker.submit_alpaca_stop_limit_order",
            return_value=self._broker_result("broker-stop-limit"),
        ) as stop_limit_mock, patch(
            "trade_scout.execution.broker.submit_alpaca_stop_order",
            side_effect=AssertionError("plain stop should not be used for crypto"),
        ):
            result = submit_stop_order(
                client_order_id="req-crypto:STOP",
                market_type="crypto",
                symbol="BTC/USD",
                side="SHORT",
                qty=0.25,
                stop_price=100.0,
            )

        self.assertEqual("broker-stop-limit", result.broker_order_id)
        stop_limit_mock.assert_called_once_with(
            client_order_id="req-crypto:STOP",
            symbol="BTC/USD",
            side="SHORT",
            qty=0.25,
            stop_price=100.0,
            limit_price=99.9,
            time_in_force="gtc",
        )

    def test_submit_stop_order_keeps_plain_stop_for_non_crypto(self) -> None:
        with patch(
            "trade_scout.execution.broker.submit_alpaca_stop_order",
            return_value=self._broker_result("broker-stop"),
        ) as stop_mock, patch(
            "trade_scout.execution.broker.submit_alpaca_stop_limit_order",
            side_effect=AssertionError("stop-limit should not be used for non-crypto"),
        ):
            result = submit_stop_order(
                client_order_id="req-stocks:STOP",
                market_type="stocks",
                symbol="AAPL",
                side="SHORT",
                qty=1.0,
                stop_price=95.0,
            )

        self.assertEqual("broker-stop", result.broker_order_id)
        stop_mock.assert_called_once_with(
            client_order_id="req-stocks:STOP",
            symbol="AAPL",
            side="SHORT",
            qty=1.0,
            stop_price=95.0,
        )

    @staticmethod
    def _broker_result(broker_order_id: str) -> BrokerOrderResult:
        return BrokerOrderResult(
            broker="alpaca_paper",
            broker_order_id=broker_order_id,
            submitted_ts_utc="2026-03-29T12:00:00+00:00",
            raw_response_json='{"id":"%s"}' % broker_order_id,
            broker_status_raw="submitted",
            accepted_ts_utc="2026-03-29T12:00:01+00:00",
            canceled_ts_utc=None,
            expired_ts_utc=None,
            rejected_ts_utc=None,
            filled_ts_utc=None,
            filled_qty=None,
            filled_avg_price=None,
            status_reason=None,
            last_broker_update_ts_utc="2026-03-29T12:00:01+00:00",
        )

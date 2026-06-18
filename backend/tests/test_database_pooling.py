import unittest

from core.database import PooledConnection


class FakeConnection:
    def __init__(self, events):
        self.events = events
        self.autocommit = False
        self.closed = False

    def __enter__(self):
        self.events.append("enter")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.events.append("rollback" if exc_type else "commit")

    def rollback(self):
        self.events.append("rollback")


class FakePool:
    def __init__(self, events):
        self.events = events

    def putconn(self, conn):
        self.events.append("putconn")


class PooledConnectionTests(unittest.TestCase):
    def test_context_manager_commits_before_returning_connection_to_pool(self):
        events = []
        conn = PooledConnection(FakePool(events), FakeConnection(events))

        with conn:
            pass

        self.assertEqual(events, ["enter", "commit", "putconn"])

    def test_plain_close_rolls_back_uncommitted_transaction_before_pool_return(self):
        events = []
        conn = PooledConnection(FakePool(events), FakeConnection(events))

        conn.close()

        self.assertEqual(events, ["rollback", "putconn"])


if __name__ == "__main__":
    unittest.main()

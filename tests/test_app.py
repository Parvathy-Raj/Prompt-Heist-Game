from arena.backends import MockBackend
from arena.db import DB


def test_rate_limiter_per_minute_and_daily():
    from app import RateLimiter
    rl = RateLimiter(per_min=2, per_day=3)
    assert rl.allow("c", 0)[0] and rl.allow("c", 1)[0]
    assert not rl.allow("c", 2)[0]          # 3rd in the same minute
    assert rl.allow("other", 2)[0]          # independent clients
    assert rl.allow("c", 100)[0]            # new minute
    ok, why = rl.allow("c", 200)
    assert not ok and "Daily" in why        # 4th of the day


def test_app_builds(tmp_path):
    from app import make_app
    demo = make_app(MockBackend(), DB(tmp_path / "a.sqlite3"))
    assert demo is not None

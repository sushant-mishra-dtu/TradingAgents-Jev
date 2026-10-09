"""Sync jobs against a fake archive, the polite client, and the CLI.

The fake archive serves the real fixture slices at their real URLs and 404s
everything else, as the archive does on a holiday. The client's throttle,
retries and 403/429 handling run against a fake session and a fake clock.
"""

import os
import subprocess
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
import requests
from requests.adapters import HTTPAdapter
from typer.testing import CliRunner

import tradingagents.dataflows.config as config_module
from cli.main import app
from tradingagents.dataflows.vendors.india import nse, store, sync
from tradingagents.dataflows.vendors.india.fetch import ArchiveClient, FetchFailed, SourceBlocked

pytestmark = pytest.mark.unit
FIXTURES = Path(__file__).parent / "fixtures" / "india"
OCT1, JUL5 = date(2026, 10, 1), date(2024, 7, 5)


def archive() -> dict[str, bytes]:
    f = lambda name: (FIXTURES / name).read_bytes()  # noqa: E731
    return {
        nse.equity_list_url(): f("EQUITY_L.csv"),
        **{nse.index_url(i): f("ind_nifty50list.csv") for i in nse.INDEX_FILES},
        nse.udiff_bhav_url(OCT1): f("BhavCopy_NSE_CM_0_0_0_20261001_F_0000.csv.zip"),
        nse.mto_url(OCT1): f("MTO_01102026.DAT"),
        nse.cm_bhav_url(JUL5): f("cm05JUL2024bhav.csv.zip"),
        nse.pr_url(OCT1): f("PR011026.zip"),
        nse.pr_url(date(2015, 1, 2)): f("PR020115.zip"),
    }


class FakeArchive(ArchiveClient):
    def __init__(self, raw, files=None, fail=None):
        super().__init__(raw, interval=0, sleep=lambda s: None)
        self.files = archive() if files is None else files
        self.fail = fail
        self.asked: list[str] = []

    def _download(self, url):
        self.asked.append(url)
        self.requests += 1
        if self.fail:
            raise self.fail
        return self.files.get(url)


@pytest.fixture
def conn(tmp_path):
    c = store.connect(tmp_path / "india.db")
    yield c
    c.close()


@pytest.fixture
def make(conn, tmp_path, monkeypatch):
    monkeypatch.setattr(sync, "MIN_LISTED", 1)

    def build(**kw):
        return sync.Syncer(conn, client=FakeArchive(tmp_path / "raw", **kw), today=date(2026, 10, 5))
    return build


def test_securities_come_from_the_equity_list_with_industries(make, conn):
    result = make().sync_securities()
    assert result.done == 1
    row = dict(conn.execute("SELECT * FROM securities WHERE nse_symbol='RELIANCE'").fetchone())
    assert row["isin"] == "INE002A01018" and row["status"] == "listed" and row["face_value"] == 10.0
    assert row["industry"] == "Oil Gas & Consumable Fuels"


def test_a_short_equity_list_is_refused_rather_than_unlisting_the_market(make, monkeypatch):
    monkeypatch.setattr(sync, "MIN_LISTED", 1000)
    with pytest.raises(sync.SyncAborted, match="refusing"):
        make().sync_securities()


def test_prices_with_deliveries_and_holidays(make, conn):
    s = make()
    s.sync_securities()
    result = s.sync_prices(date(2026, 9, 28), date(2026, 10, 2))
    assert (result.done, result.missing, result.failed) == (1, 4, 0)
    row = dict(conn.execute("SELECT * FROM prices_daily WHERE isin='INE002A01018'").fetchone())
    assert (row["date"], row["close"], row["volume"], row["deliverable_qty"]) == (
        "2026-10-01", 1167.7, 16771221, 10270423)
    assert row["source"] == "NSE bhavcopy (UDiFF)"
    statuses = dict(conn.execute("SELECT key, status FROM ingest_log WHERE job='prices'").fetchall())
    assert statuses["2026-09-28"] == "missing" and statuses["2026-10-01"] == "ok"


def test_a_sync_resumes_and_force_redoes(make):
    s = make()
    s.sync_prices(date(2026, 9, 30), date(2026, 10, 1))
    again = s.sync_prices(date(2026, 9, 30), date(2026, 10, 1))
    assert again.skipped == 2 and again.done == 0 and s.client.asked.count(nse.udiff_bhav_url(OCT1)) == 1
    s.force = True
    forced = s.sync_prices(date(2026, 9, 30), date(2026, 10, 1))
    assert forced.done == 1  # read again from the raw cache: no second download
    assert s.client.asked.count(nse.udiff_bhav_url(OCT1)) == 1


def test_old_bhavcopies_teach_which_isin_a_symbol_meant(make, conn):
    make().sync_prices(JUL5, JUL5)
    assert store.symbol_map(conn, "2024-07-05")["TCS"] == "INE467B01029"


def test_pre_2016_prices_come_from_the_pr_file_by_symbol(make, conn):
    s = make()
    s.sync_securities()
    result = s.sync_prices(date(2015, 1, 2), date(2015, 1, 2))
    assert result.done == 1
    sources = {r[0] for r in conn.execute("SELECT source FROM prices_daily WHERE date='2015-01-02'")}
    assert sources == {"NSE PR file"}


def test_corporate_actions_and_shares_from_the_pr_file(make, conn):
    s = make()
    s.sync_securities()
    store.upsert_securities(conn, [{"isin": "INE000X01010", "nse_symbol": "MOLDTKPAC", "face_value": 5.0}])
    result = s.sync_actions(OCT1, OCT1)
    assert result.done == 1
    bonus = dict(conn.execute("SELECT * FROM corporate_actions WHERE isin='INE000X01010'").fetchone())
    assert (bonus["type"], bonus["factor"], bonus["ex_date"], bonus["first_seen"]) == ("bonus", 2.0, "2026-10-09",
                                                                                          "2026-10-01")
    shares = store.get_shares_outstanding("INE144J01027", conn=conn)  # 20MICRONS
    assert shares["shares"] == 35286502


def test_announcements_for_a_universe_only(make, conn):
    s = make()
    store.upsert_securities(conn, [{"isin": "INE000Y01010", "nse_symbol": "MAHICKRA"},
                                   {"isin": "INE000Y01020", "nse_symbol": "MODIRUBBER"}])
    s.sync_announcements(OCT1, OCT1, {"INE000Y01010"}, "test")
    docs = store.get_documents("INE000Y01010", conn=conn)
    assert docs and docs[0]["kind"] == "announcement" and "symbol=MAHICKRA" in docs[0]["url"]
    assert store.get_documents("INE000Y01020", conn=conn) == []


@pytest.mark.parametrize("category, text, kind", [
    ("Credit Rating", "x", "credit_rating"),
    ("Analysts/Institutional Investor Meet/Con. Call Updates", "x", "concall"),
    (None, "Transcript of the earnings call held on", "concall"),
    ("Investor Presentation", "x", "investor_presentation"),
    ("Annual Report", "x", "annual_report"),
    ("Outcome of Board Meeting", "x", "board_meeting"),
    ("Trading Window", "x", None),
    ("Updates", "x", "announcement"),
])
def test_announcements_are_classified(category, text, kind):
    assert sync.classify_announcement(category, text) == kind


def test_a_refusing_host_stops_the_run(make, conn):
    s = make(fail=SourceBlocked("archives.nseindia.com refuses this client"))
    with pytest.raises(sync.SyncAborted, match="refuses"):
        s.sync_prices(OCT1, OCT1)
    assert conn.execute("SELECT status FROM ingest_log WHERE job='prices'").fetchone()[0] == "failed"


def test_days_logged_missing_just_before_a_refusal_are_retried_next_run(make, conn):
    # 2 October has no file here: in a real run that 403 may have been the start of the
    # refusal, so once 5 October finds the host refusing us, 2 October is failed too.
    s = make()
    s.sync_securities()
    blocked = {url for _, url in nse.price_sources(date(2026, 10, 5))}
    download = s.client._download

    def refusing(url):
        if url in blocked:
            raise SourceBlocked("archives.nseindia.com refuses this client")
        return download(url)
    s.client._download = refusing
    with pytest.raises(sync.SyncAborted, match="refuses"):
        s.sync_prices(OCT1, date(2026, 10, 5))
    statuses = dict(conn.execute("SELECT key, status FROM ingest_log WHERE job='prices'").fetchall())
    assert statuses == {"2026-10-01": "ok", "2026-10-02": "failed", "2026-10-05": "failed"}
    assert not store.is_done(conn, "prices", "2026-10-02")


def test_one_failed_day_is_logged_and_the_run_goes_on_until_failures_pile_up(make):
    s = make(fail=FetchFailed("timeout"))
    with pytest.raises(sync.SyncAborted, match="in a row"):
        s.sync_prices(date(2026, 9, 1), date(2026, 9, 30))
    s = make(fail=FetchFailed("timeout"))
    result = s.sync_prices(date(2026, 9, 28), date(2026, 9, 29))
    assert result.failed == 2 and len(result.errors) == 2


def test_import_continues_past_a_bad_file_and_logs_it(make, conn, tmp_path):
    folder = tmp_path / "inbox"
    folder.mkdir()
    for f in FIXTURES.glob("*.xml"):
        (folder / f.name).write_bytes(f.read_bytes())
    (folder / "broken.xml").write_text("<not xbrl", encoding="utf-8")
    result = make().import_files([folder])
    assert result.done == 5 and result.failed == 1
    assert "broken.xml" in result.errors[0]


def test_old_format_filings_wait_for_their_company_whatever_the_order(make, conn):
    """INDAS_* sorts before INTEGRATED_* and names its company by symbol only."""
    result = make().import_files([FIXTURES])
    assert result.failed == 0
    assert {r[0] for r in conn.execute("SELECT basis FROM filings WHERE isin='INE999Z01019' AND kind='results'")} == {
        "standalone", "consolidated"}


def test_selectors_filter_imports(make, conn):
    s = make()
    s.import_files([FIXTURES])
    s.force = True
    picked = s.import_files([FIXTURES], kinds={"shareholding"}, isins={"INE999Z01019"}, from_year=2026)
    assert picked.done == 1 and picked.skipped == 4


def test_a_forced_reimport_counts_unknown_tags_once(make, conn):
    one = FIXTURES / "INTEGRATED_FILING_INDAS_1000001_24042026105714_WEB.xml"
    s = make()
    s.import_files([one])
    s.force = True
    for _ in range(2):
        assert s.import_files([one]).done == 1
    counts = conn.execute("SELECT count FROM xbrl_unknown_tags WHERE tag='SomeNewlyIntroducedMetric'")
    assert [r[0] for r in counts] == [1]


def test_the_nightly_run_starts_after_each_jobs_last_day(make, conn, tmp_path):
    s = make()
    s.sync_prices(date(2026, 9, 28), date(2026, 10, 1))
    assert s.last_day("prices") == OCT1
    results = s.sync_all(universe="nifty50", inbox=tmp_path / "absent")
    by_job = {r.job: r for r in results}
    assert by_job["prices"].done + by_job["prices"].missing + by_job["prices"].failed == 2  # 2 and 5 October


# --- The client -------------------------------------------------------------------------

class Session:
    def __init__(self, script):
        self.script, self.headers, self.calls, self.kwargs, self.closed = list(script), {}, [], [], []

    def get(self, url, timeout, **kwargs):
        self.calls.append(url)
        self.kwargs.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        status, body, headers = item if len(item) == 3 else (*item, {})
        return SimpleNamespace(status_code=status, content=body, headers=headers,
                               close=lambda: self.closed.append(url))


def client(tmp_path, script, **kw):
    clock = [0.0]
    slept = []

    def sleep(s):
        slept.append(s)
        clock[0] += s
    c = ArchiveClient(tmp_path, interval=1.0, session=Session(script), sleep=sleep, clock=lambda: clock[0], **kw)
    return c, slept


def test_requests_are_throttled_per_host_and_identify_the_caller(tmp_path):
    c, slept = client(tmp_path, [(200, b"a"), (200, b"b")])
    c.get("https://archives.nseindia.com/a.csv", "a.csv")
    c.get("https://archives.nseindia.com/b.csv", "b.csv")
    assert slept == [1.0]
    assert c.session.headers["User-Agent"].startswith("TradingAgents/")


def test_each_request_reaches_the_wire_a_second_after_the_previous_answer_ended(tmp_path):
    # A real requests.Session, so its own preparation runs between the client's wait and the
    # hand-off. The adapter takes that preparation (slowest first, as the live smoke B4 saw) and
    # each answer's time from the fake clock, and records when each request reached it.
    clock, handed, ended = [0.0], [], []
    script = [(200, 0.001, 0.05),  # a.csv
              (403, 0.0004, 0.8), (200, 0.0006, 0.3),  # old.zip, then the canary
              (503, 0.0005, 0.05), (200, 0.0004, 0.1),  # b.csv, retried
              *[(requests.ConnectionError(), 0.0004, 0.0)] * 3,  # c.csv fails at once, three times
              (404, 0.0004, 0.2)]  # d.csv

    class Adapter(HTTPAdapter):
        def send(self, request, **kwargs):
            outcome, prep, answer = script.pop(0)
            clock[0] += prep
            handed.append(clock[0])
            clock[0] += answer
            ended.append(clock[0])
            if isinstance(outcome, Exception):
                raise outcome
            response = requests.Response()
            response.status_code, response.url, response.request = outcome, request.url, request
            response._content, response._content_consumed = b"x", True
            return response

    session = requests.Session()
    session.mount("https://", Adapter())
    c = ArchiveClient(tmp_path, session=session, clock=lambda: clock[0],
                      sleep=lambda s: clock.__setitem__(0, clock[0] + s))
    assert c.get("https://archives.nseindia.com/a.csv", "a.csv") == b"x"
    assert c.get("https://archives.nseindia.com/old.zip", "old.zip") is None
    assert c.get("https://archives.nseindia.com/b.csv", "b.csv") == b"x"
    with pytest.raises(FetchFailed):
        c.get("https://archives.nseindia.com/c.csv", "c.csv")
    assert c.get("https://archives.nseindia.com/d.csv", "d.csv") is None
    assert script == [] and len(handed) == 9
    assert min(b - a for a, b in zip(handed[:-1], handed[1:], strict=True)) >= 1.0
    assert all(next_ - end >= 1.0 for end, next_ in zip(ended[:-1], handed[1:], strict=True))


def test_the_interval_never_drops_below_one_second(tmp_path):
    assert ArchiveClient(tmp_path, interval=0.2).interval == 1.0
    assert ArchiveClient(tmp_path, interval=0).interval == 1.0
    assert ArchiveClient(tmp_path, interval=5).interval == 5.0
    config_module._config["india_request_interval"] = 0.2
    assert sync.default_client().interval == 1.0


def test_a_configured_user_agent_is_a_contact_added_to_ours(tmp_path):
    ua = ArchiveClient(tmp_path, user_agent="me@example.com").session.headers["User-Agent"]
    assert ua.startswith("TradingAgents/") and ua.endswith("contact: me@example.com")
    browser = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/129.0 Safari/537.36"
    assert ArchiveClient(tmp_path, user_agent=browser).session.headers["User-Agent"].startswith("TradingAgents/")
    sneaky = ArchiveClient(tmp_path, user_agent="me@example.com\r\nX-Forwarded-For: 1.2.3.4\x00" + "x" * 500)
    ua = sneaky.session.headers["User-Agent"]
    assert not any(ord(ch) < 32 or ord(ch) == 127 for ch in ua)
    assert len(ua.split(" contact: ", 1)[1]) == 200
    config_module._config["india_user_agent"] = "me@example.com"
    assert sync.default_client().session.headers["User-Agent"].endswith("contact: me@example.com")


def test_a_redirect_off_the_archive_host_stops_the_run_unfollowed(tmp_path):
    source = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    target = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    c, _ = client(tmp_path, [(302, b"", {"Location": target})])
    with pytest.raises(SourceBlocked) as caught:
        c.get(source, "e.csv")
    assert source in str(caught.value) and target in str(caught.value)
    assert c.session.calls == [source]


def test_a_redirect_within_the_host_fails_the_file_unfollowed(tmp_path):
    c, _ = client(tmp_path, [(301, b"", {"Location": "/content/moved/x.csv"})])
    with pytest.raises(FetchFailed, match="/content/moved/x.csv"):
        c.get("https://archives.nseindia.com/content/x.csv", "x.csv")
    assert c.session.calls == ["https://archives.nseindia.com/content/x.csv"] and not (tmp_path / "x.csv").exists()


def test_no_request_follows_a_redirect(tmp_path):
    c, _ = client(tmp_path, [(200, b"a"), (403, b"Access Denied"), (200, b"equity list"), (404, b"")])
    c.get("https://archives.nseindia.com/a.csv", "a.csv")
    c.get("https://archives.nseindia.com/old.zip", "old.zip")
    c.get("https://archives.nseindia.com/gone.zip", "gone.zip")
    assert len(c.session.kwargs) == 4
    assert all(kw.get("allow_redirects") is False for kw in c.session.kwargs)


@pytest.mark.parametrize("url", ["https://www.nseindia.com/api/x", "https://nsearchives.nseindia.com/x",
                                 "https://www.bseindia.com/x", "https://archives.nseindia.com.example.net/x"])
def test_only_the_archive_host_is_ever_asked(tmp_path, url):
    c, _ = client(tmp_path, [(200, b"never sent")])
    with pytest.raises(SourceBlocked, match="not an allowed archive host"):
        c.get(url, "x")
    assert c.session.calls == [] and c.requests == 0


def test_the_canary_is_asked_once_for_a_few_403s_and_never_downloaded(tmp_path):
    c, _ = client(tmp_path, [(403, b"Access Denied"), (200, b"equity list"), (403, b"Access Denied"),
                             (403, b"Access Denied")])
    for name in ("a.zip", "b.zip", "c.zip"):
        assert c.get(f"https://archives.nseindia.com/old/{name}", name) is None
    assert c.session.calls.count(c.canary_url) == 1 and len(c.session.calls) == 4
    canary = c.session.kwargs[c.session.calls.index(c.canary_url)]
    assert canary["stream"] is True and c.session.closed == [c.canary_url]


def test_a_cached_file_is_never_fetched_again(tmp_path):
    c, _ = client(tmp_path, [(200, b"data")])
    assert c.get("https://archives.nseindia.com/a.csv", "nse/a.csv") == b"data"
    assert c.get("https://archives.nseindia.com/a.csv", "nse/a.csv") == b"data"
    assert len(c.session.calls) == 1 and (tmp_path / "nse" / "a.csv").read_bytes() == b"data"


def test_404_is_a_missing_file(tmp_path):
    c, _ = client(tmp_path, [(404, b"")])
    assert c.get("https://archives.nseindia.com/x", "x") is None


def test_403_is_missing_when_the_host_still_serves_others(tmp_path):
    c, _ = client(tmp_path, [(403, b"Access Denied"), (200, b"equity list")])
    assert c.get("https://archives.nseindia.com/old.zip", "old.zip") is None


def test_403_everywhere_means_the_host_refuses_us(tmp_path):
    c, _ = client(tmp_path, [(403, b"Access Denied"), (403, b"Access Denied")])
    with pytest.raises(SourceBlocked):
        c.get("https://archives.nseindia.com/x", "x")


def test_the_canary_is_asked_again_after_a_run_of_403s_and_a_refusal_stops_the_run(tmp_path):
    denied = (403, b"Access Denied")
    c, _ = client(tmp_path, [denied, (200, b"equity list"), *[denied] * 4, denied, denied])
    for n in range(5):
        assert c.get(f"https://archives.nseindia.com/old/{n}.zip", f"{n}.zip") is None
    with pytest.raises(SourceBlocked, match="refuses this client"):
        c.get("https://archives.nseindia.com/old/5.zip", "5.zip")
    assert c.session.calls.count(c.canary_url) == 2 and c.session.calls[-1] == c.canary_url


def test_a_file_served_between_403s_restarts_the_count(tmp_path):
    denied = (403, b"Access Denied")
    c, _ = client(tmp_path, [denied, (200, b"equity list"), *[denied] * 3, (404, b""), *[denied] * 4,
                             (200, b"new"), *[denied] * 4])
    for n in range(14):
        c.get(f"https://archives.nseindia.com/f/{n}.zip", f"{n}.zip")
    assert c.session.calls.count(c.canary_url) == 1 and c.session.script == []


def test_429_waits_as_told_then_gives_up(tmp_path):
    c, slept = client(tmp_path, [(429, b"", {"Retry-After": "7"}), (200, b"ok")])
    assert c.get("https://archives.nseindia.com/x", "x") == b"ok" and 7.0 in slept
    c, _ = client(tmp_path, [(429, b"", {"Retry-After": "3600"})])
    with pytest.raises(SourceBlocked):
        c.get("https://archives.nseindia.com/y", "y")


def test_timeouts_and_server_errors_are_retried_with_growing_pauses(tmp_path):
    c, slept = client(tmp_path, [requests.Timeout(), (503, b""), (200, b"ok")])
    assert c.get("https://archives.nseindia.com/x", "x") == b"ok"
    assert [s for s in slept if s >= 2] == [2.0, 4.0]
    c, _ = client(tmp_path, [requests.ConnectionError()] * 3)
    with pytest.raises(FetchFailed):
        c.get("https://archives.nseindia.com/z", "z")


def test_cache_paths_cannot_escape_the_raw_directory(tmp_path):
    c, _ = client(tmp_path, [])
    with pytest.raises(ValueError):
        c.path("../../outside.txt")


# --- The CLI ------------------------------------------------------------------------------

@pytest.fixture
def cli_db(tmp_path):
    config_module._config["india_db_path"] = str(tmp_path / "cli" / "india.db")
    config_module._config["data_cache_dir"] = str(tmp_path / "cache")
    return tmp_path / "cli" / "india.db"


def test_status_before_any_sync(cli_db):
    out = CliRunner().invoke(app, ["india", "status"])
    assert out.exit_code == 0 and "No India database" in out.output


def test_import_then_status(cli_db):
    runner = CliRunner()
    out = runner.invoke(app, ["india", "import", str(FIXTURES)])
    assert out.exit_code == 0, out.output
    out = runner.invoke(app, ["india", "status"])
    assert out.exit_code == 0 and "financials" in out.output and "SomeNewlyIntroducedMetric" in out.output


def test_a_bad_date_is_a_usage_error(cli_db):
    out = CliRunner().invoke(app, ["india", "sync-prices", "--from", "June"])
    assert out.exit_code == 2


def test_a_sync_redirected_off_the_archive_exits_non_zero_naming_the_target(cli_db, monkeypatch):
    target = "https://nsearchives.nseindia.com/content/cm/moved.csv.zip"
    session = Session([(302, b"", {"Location": target})])
    monkeypatch.setattr(sync, "default_client", lambda: ArchiveClient(
        sync.raw_dir(), session=session, sleep=lambda s: None))
    out = CliRunner().invoke(app, ["india", "sync-prices", "--from", "2026-10-01", "--to", "2026-10-01"])
    assert out.exit_code == 1 and target in " ".join(out.output.split())
    assert [urlsplit(u).hostname for u in session.calls] == ["archives.nseindia.com"]


def test_a_second_sync_while_the_lock_is_held_stops_with_the_holders_pid(cli_db):
    lock = sync.raw_dir() / ".sync.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text(str(os.getpid()), encoding="ascii")  # a live process
    out = CliRunner().invoke(app, ["india", "import", str(FIXTURES)])
    assert out.exit_code == 1 and f"PID {os.getpid()}" in " ".join(out.output.split())
    assert lock.read_text(encoding="ascii") == str(os.getpid())  # still the holder's


def test_a_lock_left_by_a_dead_sync_is_taken_over_and_released(cli_db):
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    assert not sync._alive(dead.pid) and sync._alive(os.getpid())
    lock = sync.raw_dir() / ".sync.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text(str(dead.pid), encoding="ascii")
    out = CliRunner().invoke(app, ["india", "import", str(FIXTURES)])
    assert out.exit_code == 0, out.output
    assert not lock.exists()


def flat(text: str) -> str:
    """Rich wraps at the terminal width; compare with the whitespace collapsed."""
    return " ".join(text.split())


def test_importing_a_missing_path_exits_non_zero_naming_it(cli_db, tmp_path):
    missing = tmp_path / "no" / "such" / "folder"
    out = CliRunner().invoke(app, ["india", "import", str(missing)])
    assert out.exit_code == 1
    assert flat(f"no such file or folder: {missing}") in flat(out.output)


def test_importing_a_folder_without_xml_files_says_so(cli_db, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "notes.txt").write_text("not a filing", encoding="utf-8")
    out = CliRunner().invoke(app, ["india", "import", str(empty)])
    assert out.exit_code == 0, out.output
    assert flat(f"no .xml files in {empty}") in flat(out.output)


def test_a_missing_inbox_exits_non_zero_with_the_reason(cli_db, tmp_path):
    out = CliRunner().invoke(app, ["india", "sync-results", "--dir", str(tmp_path / "nowhere")])
    assert out.exit_code == 1 and "NSE serves them to browsers only" in " ".join(out.output.split())

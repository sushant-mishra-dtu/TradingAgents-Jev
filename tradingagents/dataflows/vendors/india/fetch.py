"""Polite downloads from the exchanges' public archives, cached for good.

Only archives.nseindia.com is ever asked (``ALLOWED_HOSTS``), and a redirect is
never followed: one to another host stops the run, one within the host fails
that file. Every request waits its turn per host (one a second, or longer with
TRADINGAGENTS_INDIA_REQUEST_INTERVAL; never shorter), says who is asking
(TradingAgents and its version, always; TRADINGAGENTS_INDIA_USER_AGENT adds your
contact to it), retries a timeout or a server error with growing pauses, and
honours Retry-After.

Nothing here works around an access control. archives.nseindia.com answers a
path it does not serve (bhavcopies before 2016, say) with the same Akamai
"Access Denied" page it would use to refuse a client, so the first 403 is
checked against a file the host always serves: if that is refused too, the host
is refusing us and ``SourceBlocked`` stops the run; if not, that file is
reported missing, and so are later 403s until five come in a row with nothing
served between them: then the canary is asked again, so a host that starts
refusing us mid-run still stops it. A 429 that outlasts its Retry-After also
stops the run.

Each download is written once under ``<data_cache_dir>/india/raw/`` and read
from there afterwards, so re-parsing never fetches again.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import requests

from tradingagents.dataflows.errors import VendorError

logger = logging.getLogger(__name__)

PROJECT_URL = "https://github.com/sushant-mishra-dtu/TradingAgents-Jev"
CANARY_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
# The only hosts the India layer may fetch from. NSE's www and nsearchives hosts
# and BSE's sit behind bot protection meant for browsers: never ask them.
ALLOWED_HOSTS = frozenset({"archives.nseindia.com"})
MIN_INTERVAL = 1.0  # seconds between requests to a host; a setting can only lengthen it
_MAX_CONTACT = 200
_RETRIES = 3
_MAX_RETRY_AFTER = 300.0
CANARY_RECHECK = 5  # 403s in a row taken as "not this file" before the canary is asked again


class SourceBlocked(VendorError):
    """The host is refusing this client (403 on a file it always serves, or 429
    that will not clear), or a URL or redirect points off the allowed archive
    host. Systemic: the run stops rather than keep asking."""


class FetchFailed(VendorError):
    """One file could not be fetched after retries; the run logs it and goes on."""


def default_user_agent() -> str:
    try:
        version = metadata.version("tradingagents")
    except metadata.PackageNotFoundError:
        version = "dev"
    return f"TradingAgents/{version} (India data layer; +{PROJECT_URL})"


def user_agent_with(contact: str | None) -> str:
    """``default_user_agent()``, with ``contact`` (an email or a URL) appended:
    a setting can say how to reach you, never pretend to be someone else."""
    contact = "".join(ch for ch in contact or "" if ch.isprintable()).strip()[:_MAX_CONTACT].strip()
    return f"{default_user_agent()} contact: {contact}" if contact else default_user_agent()


class ArchiveClient:
    """GETs files from the exchanges' archives: throttled, retried, cached."""

    def __init__(self, raw_dir: str | Path, *, interval: float = 1.0, user_agent: str | None = None,
                 timeout: float = 30.0, session: requests.Session | None = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
                 canary_url: str = CANARY_URL):
        self.raw_dir = Path(raw_dir)
        self.interval = max(MIN_INTERVAL, float(interval))
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent_with(user_agent),
                                     "Accept": "*/*", "Accept-Language": "en-IN,en;q=0.8"})
        self.sleep, self.clock = sleep, clock
        self.canary_url = canary_url
        self._last: dict[str, float] = {}
        self._refusals: dict[str, bool] = {}  # host -> the canary's latest verdict
        self._unchecked: dict[str, int] = {}  # host -> 403s taken as missing since a file was served
        self.requests = 0
        self.downloaded = 0  # bytes

    # Cache ---------------------------------------------------------------------
    def path(self, relative: str) -> Path:
        path = (self.raw_dir / relative).resolve()
        if self.raw_dir.resolve() not in path.parents:
            raise ValueError(f"cache path escapes the raw directory: {relative!r}")
        return path

    def cached(self, relative: str) -> bytes | None:
        path = self.path(relative)
        return path.read_bytes() if path.is_file() else None

    def store(self, relative: str, data: bytes) -> Path:
        path = self.path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        temp.write_bytes(data)
        os.replace(temp, path)
        return path

    # Network -------------------------------------------------------------------
    def get(self, url: str, cache_as: str, *, refresh: bool = False) -> bytes | None:
        """The file at ``url``, from the cache when it is there. None when the
        archive has no such file (404, or a path the host does not serve)."""
        if not refresh and (data := self.cached(cache_as)) is not None:
            return data
        data = self._download(url)
        if data is not None:
            self.store(cache_as, data)
        return data

    def _wait(self, host: str) -> None:
        last = self._last.get(host)
        if last is not None:
            remaining = self.interval - (self.clock() - last)
            if remaining > 0:
                self.sleep(remaining)
        self._last[host] = self.clock()

    def _request(self, url: str, **kwargs) -> requests.Response:
        host = urlsplit(url).hostname
        if host not in ALLOWED_HOSTS:
            raise SourceBlocked(f"refusing to fetch {host}: not an allowed archive host")
        self._wait(urlsplit(url).netloc)
        self.requests += 1
        return self.session.get(url, timeout=self.timeout, allow_redirects=False, **kwargs)

    def _download(self, url: str) -> bytes | None:
        pause = 2.0
        for attempt in range(1, _RETRIES + 1):
            try:
                response = self._request(url)
            except requests.RequestException as exc:
                if attempt == _RETRIES:
                    raise FetchFailed(f"{url}: {type(exc).__name__} after {attempt} tries") from exc
                self.sleep(pause)
                pause *= 2
                continue
            status = response.status_code
            if 300 <= status < 400:
                _refuse_redirect(url, status, response.headers.get("Location"))
            host = urlsplit(url).hostname
            if status == 200:
                self._unchecked[host] = 0
                self.downloaded += len(response.content)
                return response.content
            if status == 404:
                self._unchecked[host] = 0
                return None
            if status == 403:
                if self._refused(host):
                    raise SourceBlocked(f"{urlsplit(url).netloc} refuses this client (403 on {url} "
                                        f"and on {self.canary_url})")
                self._unchecked[host] = self._unchecked.get(host, 0) + 1
                logger.info("403 on %s while the host still serves others: treated as missing", url)
                return None
            if status == 429:
                wait = _retry_after(response.headers.get("Retry-After"), pause)
                if attempt == _RETRIES or wait > _MAX_RETRY_AFTER:
                    raise SourceBlocked(f"{urlsplit(url).netloc} keeps throttling (429 on {url})")
                self.sleep(wait)
                pause *= 2
                continue
            if 500 <= status < 600 and attempt < _RETRIES:
                self.sleep(pause)
                pause *= 2
                continue
            raise FetchFailed(f"{url}: HTTP {status}")
        raise FetchFailed(f"{url}: no answer after {_RETRIES} tries")

    def _refused(self, host: str | None) -> bool:
        """Whether a 403 means "not you" or "not this file": ask for a file the
        host always serves, reading only the status. A "not this file" verdict
        holds until ``CANARY_RECHECK`` 403s come in a row with no file served
        between them (old paths the host never served, say); then it is asked
        again, in case the host has started refusing us since."""
        if not self.canary_url:
            return True
        if host not in self._refusals or self._unchecked.get(host, 0) >= CANARY_RECHECK:
            try:
                response = self._request(self.canary_url, stream=True)
            except requests.RequestException:
                return True
            try:
                self._refusals[host] = response.status_code in (401, 403, 429)
            finally:
                response.close()
            self._unchecked[host] = 0
        return self._refusals[host]


def _refuse_redirect(url: str, status: int, location: str | None) -> None:
    """Redirects are never followed. One off the host stops the run: every later
    file would redirect too, and where NSE moved its files is for a person to
    check (the URLs are in india/nse.py). One within the host fails this file."""
    if not location:
        raise FetchFailed(f"{url}: HTTP {status} without a Location")
    target = urljoin(url, location)
    if urlsplit(target).hostname != urlsplit(url).hostname:
        raise SourceBlocked(f"{url} redirects (HTTP {status}) to {target}, off the archive host; not followed")
    raise FetchFailed(f"{url}: moved to {urlsplit(target).path} (HTTP {status}); not followed")


def _retry_after(value: str | None, default: float) -> float:
    try:
        return max(0.0, float(value)) if value else default
    except ValueError:
        return default

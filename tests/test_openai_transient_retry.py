"""Transient provider errors are retried with a slow backoff.

Shared/trial endpoints (NVIDIA NIM) return bursts of HTTP 500 mid-run that
outlast the SDK's own few-second retries; one such burst used to abort a whole
analysis at the Market Analyst's seventh tool turn.
"""
from __future__ import annotations

import httpx
import openai
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from tradingagents.llm_clients import openai_client
from tradingagents.llm_clients.openai_client import NormalizedChatOpenAI


def _server_error():
    request = httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions")
    response = httpx.Response(500, request=request)
    return openai.InternalServerError("Internal server error", response=response, body=None)


def _auth_error():
    request = httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions")
    response = httpx.Response(401, request=request)
    return openai.AuthenticationError("bad key", response=response, body=None)


@pytest.fixture
def sleeps(monkeypatch):
    recorded = []
    monkeypatch.setattr(openai_client.time, "sleep", recorded.append)
    return recorded


@pytest.fixture
def llm():
    return NormalizedChatOpenAI(model="nvidia/nemotron-3-super-120b-a12b", api_key="k")


def _script(monkeypatch, outcomes):
    calls = []

    def fake_invoke(self, input, config=None, **kwargs):
        calls.append(input)
        outcome = outcomes[len(calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(BaseChatModel, "invoke", fake_invoke)
    return calls


@pytest.mark.unit
def test_server_error_is_retried_until_success(llm, sleeps, monkeypatch):
    calls = _script(monkeypatch, [_server_error(), _server_error(), AIMessage(content="ok")])
    assert llm.invoke("hi").content == "ok"
    assert len(calls) == 3
    assert sleeps == [10, 20]


@pytest.mark.unit
def test_gives_up_after_backoff_is_exhausted(llm, sleeps, monkeypatch):
    attempts = len(openai_client._TRANSIENT_BACKOFF_SECONDS) + 1
    calls = _script(monkeypatch, [_server_error()] * attempts)
    with pytest.raises(openai.InternalServerError):
        llm.invoke("hi")
    assert len(calls) == attempts
    assert sleeps == list(openai_client._TRANSIENT_BACKOFF_SECONDS)


@pytest.mark.unit
def test_non_transient_error_is_not_retried(llm, sleeps, monkeypatch):
    calls = _script(monkeypatch, [_auth_error()])
    with pytest.raises(openai.AuthenticationError):
        llm.invoke("hi")
    assert len(calls) == 1
    assert sleeps == []

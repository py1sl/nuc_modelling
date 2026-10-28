"""Tests for the IAEA LiveChart data helpers."""

from io import BytesIO
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from nuc_modelling.nuclear_data import (
    iaea_livechart_helper as livechart_helper,
)


def test_lc_pd_dataframe_sets_user_agent_and_closes_response(monkeypatch):
    response = BytesIO(b"value\n1\n")
    request_details = {}

    def fake_urlopen(request):
        request_details["request"] = request
        return response

    monkeypatch.setattr(
        livechart_helper.urllib.request, "urlopen", fake_urlopen
    )

    result = livechart_helper.lc_pd_dataframe("https://example.test/data")

    assert result.to_dict(orient="list") == {"value": [1]}
    assert (
        request_details["request"].get_header("User-agent")
        == "Livechart/1.0"
    )
    assert response.closed


@pytest.mark.parametrize(
    ("nuc_str", "error"),
    [(None, TypeError), ("", ValueError), ("  ", ValueError)],
)
def test_get_hl_rejects_invalid_nuclide(nuc_str, error):
    with pytest.raises(error, match="nuc_str"):
        livechart_helper.get_hl(nuc_str)


def test_get_hl_encodes_nuclide_and_returns_first_value(monkeypatch):
    requested_urls = []
    monkeypatch.setattr(
        livechart_helper,
        "lc_pd_dataframe",
        lambda url: requested_urls.append(url) or pd.DataFrame(
            {"half_life_sec": [12.5, 99.0]}
        ),
    )

    result = livechart_helper.get_hl("U 235&")

    assert result == 12.5
    query = parse_qs(urlparse(requested_urls[0]).query)
    assert query == {"fields": ["ground_states"], "nuclides": ["U 235&"]}


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame(columns=["other"]), "half_life_sec"),
        (pd.DataFrame(columns=["half_life_sec"]), "No half-life data"),
    ],
)
def test_get_hl_rejects_missing_or_empty_data(monkeypatch, frame, message):
    monkeypatch.setattr(
        livechart_helper, "lc_pd_dataframe", lambda _url: frame
    )

    with pytest.raises(ValueError, match=message):
        livechart_helper.get_hl("U-235")


def test_get_gamma_for_nuc_filters_and_cleans_data(monkeypatch):
    frame = pd.DataFrame(
        {
            "p_energy": ["0", "0.5", "0", "not-a-number"],
            "intensity": ["12.5", "50", "unknown", "30"],
            "half_life": ["a", "b", "c", "d"],
            "other": [1, 2, 3, 4],
        }
    )
    requested_urls = []
    monkeypatch.setattr(
        livechart_helper,
        "lc_pd_dataframe",
        lambda url: requested_urls.append(url) or frame.copy(),
    )

    result = livechart_helper.get_gamma_for_nuc("U-235&")

    assert result["intensity"].tolist() == [12.5]
    assert result["other"].tolist() == [1]
    assert "half_life" not in result.columns
    query = parse_qs(urlparse(requested_urls[0]).query)
    assert query == {
        "fields": ["decay_rads"],
        "nuclides": ["U-235&"],
        "rad_types": ["g"],
    }


@pytest.mark.parametrize(
    ("frame", "missing"),
    [
        (pd.DataFrame({"intensity": [1]}), "p_energy"),
        (pd.DataFrame({"p_energy": [0]}), "intensity"),
    ],
)
def test_get_gamma_for_nuc_requires_response_columns(
    monkeypatch, frame, missing
):
    monkeypatch.setattr(
        livechart_helper, "lc_pd_dataframe", lambda _url: frame
    )

    with pytest.raises(ValueError, match=missing):
        livechart_helper.get_gamma_for_nuc("U-235")


def test_get_gamma_for_nuc_handles_empty_data_and_absent_optional_columns(
    monkeypatch,
):
    frame = pd.DataFrame({"p_energy": [], "intensity": []})
    monkeypatch.setattr(
        livechart_helper, "lc_pd_dataframe", lambda _url: frame
    )

    result = livechart_helper.get_gamma_for_nuc("U-235")

    assert result.empty
    assert list(result.columns) == ["p_energy", "intensity"]

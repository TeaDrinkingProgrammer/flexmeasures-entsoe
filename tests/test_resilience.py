"""
Tests for resilience against ENTSO-E failures.

Covered:

- outages treat ENTSO-E's "no matching data" as "no outages announced".

No live API and no database are touched: ENTSO-E is stood in for by a fake client.
``flexmeasures`` must be importable for these (the package __init__ pulls it in); if it
is not installed, the whole module is skipped.
"""

import pandas as pd
import pytest
import requests

pytest.importorskip("flexmeasures")

from entsoe.exceptions import NoMatchingDataError  # noqa: E402

from flexmeasures_entsoe.prices import regressors_import  # noqa: E402

TZ = "Europe/Amsterdam"
FROM_TIME = pd.Timestamp("2025-06-02 00:00", tz=TZ)
UNTIL_TIME = pd.Timestamp("2025-06-03 00:00", tz=TZ)


class FakeLog:
    def __init__(self):
        self.warnings = []

    def info(self, *args):
        pass

    def debug(self, *args):
        pass

    def warning(self, msg, *args):
        self.warnings.append(msg)


# ---------------------------------------------------------------------------
# Outages: "no matching data" means no outages were announced
# ---------------------------------------------------------------------------


class FakeOutagesClient:
    """Fake ENTSO-E client for outages.

    mode: 'ok' | 'no-data' | 'raise'
    """

    def __init__(self, mode="ok"):
        self.mode = mode

    def query_unavailability_of_generation_units(self, country_code, start, end):
        if self.mode == "no-data":
            raise NoMatchingDataError()
        if self.mode == "raise":
            raise requests.ConnectionError("connection reset")
        return pd.DataFrame(
            {
                "nominal_power": [1000.0],
                "avail_qty": [0.0],
                "start": [FROM_TIME],
                "end": [UNTIL_TIME],
            },
            index=pd.Index([FROM_TIME - pd.Timedelta(days=2)], name="created_doc_time"),
        )


def _collect_outages(client):
    results = regressors_import._collect_country_regressors(
        client=client,
        log=FakeLog(),
        country_code="NL",
        from_time=FROM_TIME,
        until_time=UNTIL_TIME,
        include_load_forecast=False,
        include_wind_solar_forecast=False,
        include_residual_load=False,
        include_outages=True,
        strict=True,
    )
    return {spec[0]: series for spec, _override, series, _is_entsoe in results}


def test_outages_without_matching_data_mean_no_outages():
    """ENTSO-E answers "no matching data" if no outages were announced: 0 MW."""
    saved = _collect_outages(FakeOutagesClient(mode="no-data"))
    assert saved["Generation outages"].to_numpy() == pytest.approx([0.0] * 24)


def test_outages_are_counted():
    """Announced outages still count as unavailable capacity."""
    saved = _collect_outages(FakeOutagesClient())
    assert saved["Generation outages"].to_numpy() == pytest.approx([1000.0] * 24)


def test_outages_query_failure_still_aborts():
    """Other failures to get outages still abort the import."""
    with pytest.raises(requests.ConnectionError):
        _collect_outages(FakeOutagesClient(mode="raise"))

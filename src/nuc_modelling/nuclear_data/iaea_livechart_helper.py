"""Helpers for retrieving nuclear data from the IAEA LiveChart service."""

from urllib.parse import urlencode
import pandas as pd
import urllib.request

# The service URL.
livechart = "https://nds.iaea.org/relnsd/v1/data?"

# There have been cases in which the service returns HTTP 403; use this
# workaround.


def lc_pd_dataframe(url):
    """Fetch a LiveChart CSV URL and return its contents as a dataframe."""
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Livechart/1.0")
    with urllib.request.urlopen(req) as response:
        return pd.read_csv(response)


def _nucleus_query(fields, nuc_str, **extra_params):
    if not isinstance(nuc_str, str):
        raise TypeError("nuc_str must be a non-empty string")
    if not nuc_str.strip():
        raise ValueError("nuc_str must be a non-empty string")

    params = {"fields": fields, "nuclides": nuc_str, **extra_params}
    return livechart + urlencode(params)


def get_hl(nuc_str):
    """Return the ground-state half-life in seconds for a nuclide."""
    df = lc_pd_dataframe(_nucleus_query("ground_states", nuc_str))
    if "half_life_sec" not in df.columns:
        raise ValueError(
            "LiveChart response is missing the 'half_life_sec' column"
        )
    if df.empty:
        raise ValueError(f"No half-life data returned for nucleus {nuc_str!r}")
    return df["half_life_sec"].iloc[0]


def get_gamma_for_nuc(nuc_str):
    """Return gamma-ray data after filtering invalid lines."""
    url = _nucleus_query("decay_rads", nuc_str, rad_types="g")
    df = lc_pd_dataframe(url)
    missing_columns = {"p_energy", "intensity"}.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"LiveChart response is missing required column(s): {missing}"
        )

    p_energy = pd.to_numeric(df["p_energy"], errors="coerce")
    intensity = pd.to_numeric(df["intensity"], errors="coerce")
    df = df.loc[(p_energy == 0) & intensity.notna()].copy()
    df["intensity"] = intensity.loc[df.index].astype(float)
    return df.drop(
        columns=[
            "half_life",
            "operator_hl",
            "unc_hl",
            "unit_hl",
            "unc_hls",
            "ensdf_authors",
            "ensdf_publication_cut-off",
            "Extraction_date",
        ],
        errors="ignore",
    )

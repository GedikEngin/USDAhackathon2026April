"""
forecast/chip_picker.py
Select T=3 chips per (GEOID, year, forecast_date) query for D.1.d's
Prithvi embedding extraction.

The chip picker is pure pandas — no rasters, no GPU. It reads the
canonical chip_index.parquet (built by scripts/merge_chip_index.py) and
returns ChipPick objects that the inference loop loads from disk.

The picker enforces the locked design from PHASE2_DECISIONS_LOG.md
entry 2-D.1.kickoff:

  - T=3 sequence per query, drawn from three phenology phases:
    vegetative (aug1), silking (sep1), grain-fill (oct1 / final).
  - As-of rule: every chip's scene_date strictly less than forecast_date.
    "08-01" -> chips with scene_date < <year>-08-01.
  - When a phase has no qualifying chip (e.g. grain-fill at 08-01), pad
    with the silking chip duplicated so the tensor shape stays (T=3).
  - Within a phase, prefer chips that pass quality gates
    (valid_pixel_frac >= 0.5 AND corn_pixel_frac >= 0.05); among those,
    pick the most recent scene_date; tiebreak by valid_pixel_frac desc,
    then corn_pixel_frac desc, then chip_path alpha (determinism).
  - If no chip passes the quality gates, fall back to the
    highest-valid_pixel_frac chip available in that phase.

Forecast-date semantics:

  forecast_date    as-of cutoff (in year)   phases sourceable from
  -------------    ----------------------   ----------------------
  "08-01"          <year>-08-01             aug1
  "09-01"          <year>-09-01             aug1, sep1
  "10-01"          <year>-10-01             aug1, sep1, oct1
  "EOS"            <year>-11-15             aug1, sep1, oct1, final

The "EOS" cutoff matches the GROWING_SEASON_END_MD in hls_common; it's
effectively "everything we've got."

Public API:

    pick_chips(index_df, geoid, year, forecast_date) -> ChipQuery | None
    pick_all(index_df, queries: Iterable[(geoid, year, forecast_date)]) -> dict

Both return ChipQuery objects with the picked chip_paths AND the QC
columns (chip_count, max scene-date distance, min corn fraction, max
cloud) which become explicit regressor features per decisions log.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Iterable, Optional

import pandas as pd


# =============================================================================
# Constants
# =============================================================================

# Quality gates per chip (decisions log: 5% corn-frac filter; valid-frac
# >= 0.5 is a softer "we have enough clear pixels to be useful" bar).
MIN_CORN_FRAC: float = 0.05
MIN_VALID_FRAC: float = 0.5

# T=3 sequence per query.
T_SEQUENCE: int = 3

# Names of the three phenology slots in the returned sequence. Order
# matters: Prithvi will see them in this order as the temporal axis.
PHENO_SLOTS: tuple[str, ...] = ("vegetative", "silking", "grain_fill")

# For each forecast_date, which raw phases (chip_index labels) are eligible
# for each phenology slot. Lists are searched in order; first non-empty
# pool is used. The "as-of cutoff" filter is applied separately to all
# phases before consulting these lists.
#
# Note for 08-01: only aug1 is available pre-cutoff; the silking and
# grain-fill slots fall back to vegetative through the padding logic.
# Same for 09-01: oct1/final not yet acquired pre-cutoff.
PHASE_POOL_BY_FORECAST_DATE: dict[str, dict[str, tuple[str, ...]]] = {
    "08-01": {
        "vegetative": ("aug1",),
        "silking":    (),
        "grain_fill": (),
    },
    "09-01": {
        "vegetative": ("aug1",),
        "silking":    ("sep1",),
        "grain_fill": (),
    },
    "10-01": {
        "vegetative": ("aug1",),
        "silking":    ("sep1",),
        "grain_fill": ("oct1",),
    },
    "EOS": {
        "vegetative": ("aug1",),
        "silking":    ("sep1",),
        "grain_fill": ("oct1", "final"),  # prefer oct1, fall back to final
    },
}

VALID_FORECAST_DATES: tuple[str, ...] = tuple(PHASE_POOL_BY_FORECAST_DATE.keys())


def forecast_date_cutoff(year: int, forecast_date: str) -> dt.date:
    """Return the as-of cutoff date as a datetime.date.

    Chips with scene_date strictly less than this date are eligible.
    """
    if forecast_date not in VALID_FORECAST_DATES:
        raise ValueError(
            f"forecast_date must be one of {VALID_FORECAST_DATES}, got {forecast_date!r}"
        )
    if forecast_date == "EOS":
        # EOS is end-of-growing-season; matches GROWING_SEASON_END_MD in
        # hls_common.py (Nov 15). Anything in the season is fair game.
        return dt.date(year, 11, 15)
    mm, dd = forecast_date.split("-")
    return dt.date(year, int(mm), int(dd))


# =============================================================================
# Result dataclasses
# =============================================================================


@dataclass
class ChipPick:
    """One pick within a sequence; can be a 'real' pick or a padded duplicate."""
    chip_path:        str
    scene_date:       dt.date
    phase:            str        # raw chip_index phase label
    pheno_slot:       str        # one of PHENO_SLOTS
    sensor:           str
    mgrs_tile:        str
    corn_pixel_frac:  float
    valid_pixel_frac: float
    is_padded:        bool       # True if this is a duplicate-pad

    def quality_passed(self) -> bool:
        return (self.valid_pixel_frac >= MIN_VALID_FRAC
                and self.corn_pixel_frac >= MIN_CORN_FRAC)


@dataclass
class ChipQuery:
    """Full result for one (GEOID, year, forecast_date) query."""
    GEOID:         str
    year:          int
    forecast_date: str
    picks:         list[ChipPick] = field(default_factory=list)

    # QC features the regressor consumes (decisions log: chip_age_days,
    # chip_count, cloud_pct_max, corn_pixel_frac_min)
    chip_count:           int = 0    # number of non-padded picks
    chip_age_days_max:    Optional[int] = None    # max(forecast_date - scene_date)
    cloud_pct_max:        Optional[float] = None  # 1 - min(valid_pixel_frac)
    corn_pixel_frac_min:  Optional[float] = None

    @property
    def chip_paths(self) -> list[str]:
        return [p.chip_path for p in self.picks]

    @property
    def has_no_real_chips(self) -> bool:
        """True iff every pick is a pad (no real chip data found)."""
        return self.chip_count == 0


# =============================================================================
# Core picker
# =============================================================================


def _filter_for_query(
    index_df: pd.DataFrame,
    geoid: str,
    year: int,
    cutoff: dt.date,
) -> pd.DataFrame:
    """Filter the index to positive (chip-written) rows for this county+year
    that satisfy the as-of rule (scene_date < cutoff)."""
    geoid = str(geoid).zfill(5)
    df = index_df
    df = df[(df["GEOID"] == geoid)
            & (df["year"] == year)
            & df["chip_path"].notna()]
    if df.empty:
        return df
    # Scene_date may have been written as object/date32 by parquet; coerce
    # to datetime64 for safe comparison.
    sd = pd.to_datetime(df["scene_date"]).dt.date
    df = df.assign(_scene_date=sd)
    df = df[df["_scene_date"] < cutoff]
    return df


def _pick_best_in_pool(
    pool_df: pd.DataFrame,
) -> Optional[pd.Series]:
    """Within a candidate pool (already filtered to one phase or a phase
    union), pick the single best chip according to:

      1. quality-gate-pass (valid_frac >= 0.5 AND corn_frac >= 0.05) preferred
      2. most-recent scene_date
      3. highest valid_pixel_frac
      4. highest corn_pixel_frac
      5. alphabetical chip_path (deterministic)

    Returns the row (Series) or None if the pool is empty.
    """
    if pool_df.empty:
        return None

    df = pool_df.copy()
    df["_quality_pass"] = (
        (df["valid_pixel_frac"] >= MIN_VALID_FRAC)
        & (df["corn_pixel_frac"] >= MIN_CORN_FRAC)
    )

    # If at least one chip passes the gates, restrict to those; else use
    # everything (fallback path).
    if df["_quality_pass"].any():
        df = df[df["_quality_pass"]]

    # Sort by composite ranking. ascending=False everywhere; chip_path tie
    # break is ascending (alpha-first wins).
    df = df.sort_values(
        by=["_scene_date", "valid_pixel_frac", "corn_pixel_frac", "chip_path"],
        ascending=[False, False, False, True],
        kind="mergesort",  # stable
    )
    return df.iloc[0]


def _row_to_chip_pick(
    row: pd.Series,
    pheno_slot: str,
    is_padded: bool,
) -> ChipPick:
    """Convert a chip_index row + slot label into a ChipPick."""
    return ChipPick(
        chip_path=str(row["chip_path"]),
        scene_date=row["_scene_date"],
        phase=str(row["phase"]),
        pheno_slot=pheno_slot,
        sensor=str(row["sensor"]),
        mgrs_tile=str(row["mgrs_tile"]),
        corn_pixel_frac=float(row["corn_pixel_frac"]),
        valid_pixel_frac=float(row["valid_pixel_frac"]),
        is_padded=is_padded,
    )


def _padded_copy(pick: ChipPick, new_slot: str) -> ChipPick:
    """Return a ChipPick that copies all fields from `pick` but is marked
    is_padded=True and assigned to `new_slot`."""
    return ChipPick(
        chip_path=pick.chip_path,
        scene_date=pick.scene_date,
        phase=pick.phase,
        pheno_slot=new_slot,
        sensor=pick.sensor,
        mgrs_tile=pick.mgrs_tile,
        corn_pixel_frac=pick.corn_pixel_frac,
        valid_pixel_frac=pick.valid_pixel_frac,
        is_padded=True,
    )


def pick_chips(
    index_df: pd.DataFrame,
    geoid: str,
    year: int,
    forecast_date: str,
) -> ChipQuery:
    """Pick T=3 chips for one (GEOID, year, forecast_date) query.

    Returns a ChipQuery. If no chips are available at all, the returned
    query has chip_count=0 and an empty picks list — D.1.d treats this
    as a missing embedding (NaN) and lets XGBoost handle via missing-
    direction split.
    """
    geoid = str(geoid).zfill(5)
    if forecast_date not in VALID_FORECAST_DATES:
        raise ValueError(
            f"forecast_date must be one of {VALID_FORECAST_DATES}, got {forecast_date!r}"
        )

    cutoff = forecast_date_cutoff(year, forecast_date)
    eligible = _filter_for_query(index_df, geoid, year, cutoff)

    pool_map = PHASE_POOL_BY_FORECAST_DATE[forecast_date]

    # Step 1: try to fill each phenology slot from its phase pool(s).
    real_picks: dict[str, Optional[ChipPick]] = {slot: None for slot in PHENO_SLOTS}
    for slot in PHENO_SLOTS:
        phase_options = pool_map[slot]
        if not phase_options:
            continue
        # Walk each phase in priority order; first non-empty pool wins.
        for phase in phase_options:
            phase_pool = eligible[eligible["phase"] == phase]
            best = _pick_best_in_pool(phase_pool)
            if best is not None:
                real_picks[slot] = _row_to_chip_pick(best, slot, is_padded=False)
                break

    # Step 2: pad empty slots from the most-recent real pick (per decisions
    # log: "T=2 padded with the silking chip duplicated"; we generalize to
    # "pad from whichever real pick has the most-recent scene_date").
    real_picks_list = [p for p in real_picks.values() if p is not None]

    if not real_picks_list:
        # No real chips at all. Return empty query — D.1.d will mark the
        # row's embedding as NaN.
        return ChipQuery(
            GEOID=geoid,
            year=int(year),
            forecast_date=forecast_date,
        )

    # Most-recent real pick = padding source
    pad_source = max(real_picks_list, key=lambda p: p.scene_date)
    final_picks: list[ChipPick] = []
    for slot in PHENO_SLOTS:
        rp = real_picks[slot]
        if rp is None:
            final_picks.append(_padded_copy(pad_source, new_slot=slot))
        else:
            final_picks.append(rp)

    # Step 3: compute QC features
    real_count = sum(1 for p in final_picks if not p.is_padded)
    real_only = [p for p in final_picks if not p.is_padded]
    chip_age_days_max = max(
        (cutoff - p.scene_date).days for p in real_only
    ) if real_only else None
    cloud_pct_max = (
        100.0 * (1.0 - min(p.valid_pixel_frac for p in real_only))
    ) if real_only else None
    corn_pixel_frac_min = min(
        p.corn_pixel_frac for p in real_only
    ) if real_only else None

    return ChipQuery(
        GEOID=geoid,
        year=int(year),
        forecast_date=forecast_date,
        picks=final_picks,
        chip_count=real_count,
        chip_age_days_max=chip_age_days_max,
        cloud_pct_max=cloud_pct_max,
        corn_pixel_frac_min=corn_pixel_frac_min,
    )


def pick_all(
    index_df: pd.DataFrame,
    queries: Iterable[tuple[str, int, str]],
) -> list[ChipQuery]:
    """Bulk pick. queries is an iterable of (GEOID, year, forecast_date).
    Returns a list of ChipQuery in the same order."""
    return [pick_chips(index_df, g, y, fd) for g, y, fd in queries]


# =============================================================================
# Diagnostic helper for the regressor pre-flight check
# =============================================================================


def coverage_summary(
    index_df: pd.DataFrame,
    queries: Iterable[tuple[str, int, str]],
) -> pd.DataFrame:
    """Run pick_chips for every query and return a DataFrame summarizing
    chip availability per query. Useful for the D.1.d preflight: how many
    queries will end up with NaN embeddings?"""
    rows = []
    for g, y, fd in queries:
        q = pick_chips(index_df, g, y, fd)
        rows.append({
            "GEOID": q.GEOID,
            "year": q.year,
            "forecast_date": q.forecast_date,
            "chip_count": q.chip_count,
            "fully_padded": q.has_no_real_chips,
            "chip_age_days_max": q.chip_age_days_max,
            "cloud_pct_max": q.cloud_pct_max,
            "corn_pixel_frac_min": q.corn_pixel_frac_min,
        })
    return pd.DataFrame(rows)

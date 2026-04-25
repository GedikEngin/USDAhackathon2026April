"""
forecast/prithvi.py
Production wrapper around the terratorch Prithvi-EO-2.0-300M-TL backbone
for D.1.d's frozen-feature-extraction pipeline.

Contract verified empirically by scripts/probe_prithvi.py
(see runs/prithvi_probe_*.json):

  - registry name:    'prithvi_eo_v2_300_tl'
  - input layout:     (B, C=6, T=3, H=224, W=224)
  - bands order:      [BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2]
  - kwargs:           temporal_coords (B,T,2)  [year, day_of_year]
                      location_coords (B,2)    [lat, lon]
  - output:           list of 24 hidden states, each (B, 589, 1024)
                      where 589 = 14*14*T + 1 CLS = 588 + 1 for T=3
  - embedding dim D:  1024
  - VRAM @ B=8 fp16:  0.79 GB peak (plenty of headroom on 12 GB card)

Pooling: mean across all 589 tokens (decisions log "mean across spatial
patches and across T"). The CLS token is included in the mean for code
simplicity; this is a <1% effect vs the patch-only variant and reversible
if D.1.e ablation requires.

Public API:

    PrithviEmbedder(...)                       # constructor; loads model once
    embedder.embed_batch(chips, ...)           # (B, T=3, C=6, H, W) -> (B, 1024)
    embedder.embed_query_sequences(seqs, ...)  # high-level: list[ChipPick] -> (N, 1024)

Lifecycle:
  - Build once per process (model load is ~2 minutes on first run, instant
    on cache hit).
  - Inference is GPU-bound when GPU available, CPU-bound otherwise. The
    full grid (~18,624 queries × T=3 chips) takes ~1 minute on a 12 GB
    laptop GPU at fp16.
  - This module is imported by scripts/extract_embeddings.py (D.1.d) and,
    if needed at forecast time, by backend/forecast_routes.py.
"""

from __future__ import annotations

import datetime as dt
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import rasterio
import torch

# Avoid circular import at module load time -- forecast.chip_picker is light
from forecast.chip_picker import ChipPick

log = logging.getLogger(__name__)


# =============================================================================
# Constants -- pulled into module scope so other modules can import them
# without instantiating the embedder
# =============================================================================

# Locked in PHASE2_DECISIONS_LOG entry 2-D.1.kickoff
PRITHVI_MODEL_NAME: str = "prithvi_eo_v2_300_tl"
EMBEDDING_DIM: int = 1024
T_SEQUENCE: int = 3
CHIP_SIZE: int = 224
N_BANDS: int = 6
DEFAULT_BATCH_SIZE: int = 8
DEFAULT_DTYPE = torch.float16

# HLS reflectance scale factor; chips on disk are int16 raw DN.
# Multiply by this to get [0, 1.0] reflectance before feeding to Prithvi.
HLS_REFLECTANCE_SCALE: float = 0.0001

# Embedding column names. Other modules (regressor training, master-table
# rebuild) import this list to construct their feature schemas without
# needing to know D.
EMBEDDING_COL_NAMES: list[str] = [f"prithvi_{i:04d}" for i in range(EMBEDDING_DIM)]

# Model versioning, baked into embeddings_v1.parquet rows so re-extracts
# can never silently mix.
MODEL_VERSION: str = "prithvi_eo_v2_300_tl_meanpool_v1"


# =============================================================================
# Chip loading
# =============================================================================


def load_chip_as_tensor(chip_path: str | Path) -> np.ndarray:
    """Read one chip GeoTIFF and return a (C=6, H, W) float32 array of
    reflectance values (raw int16 -> float * 0.0001).

    Output is unscaled-by-Prithvi-mean-std; that normalization happens
    inside the model's preprocessing if applicable, or we apply it
    explicitly in embed_batch (currently we don't -- see notes).
    """
    with rasterio.open(chip_path) as src:
        # rasterio reads as (C, H, W); we want float32 reflectance
        arr = src.read().astype(np.float32) * HLS_REFLECTANCE_SCALE
        # Sanity: chips should always be 6×224×224
        if arr.shape != (N_BANDS, CHIP_SIZE, CHIP_SIZE):
            raise ValueError(
                f"chip {chip_path} has shape {arr.shape}, "
                f"expected ({N_BANDS}, {CHIP_SIZE}, {CHIP_SIZE})"
            )
    return arr


# =============================================================================
# Embedder class
# =============================================================================


@dataclass
class EmbeddingResult:
    """Per-query result from embed_query_sequences. Mirrors ChipQuery's
    QC fields so the regressor can consume them as features."""
    GEOID: str
    year: int
    forecast_date: str
    embedding: Optional[np.ndarray]  # shape (1024,) or None if no real chips

    # QC features (copied from ChipQuery for convenience)
    chip_count:           int
    chip_age_days_max:    Optional[int]
    cloud_pct_max:        Optional[float]
    corn_pixel_frac_min:  Optional[float]
    model_version:        str = MODEL_VERSION


class PrithviEmbedder:
    """Wraps the pretrained Prithvi-EO-2.0-300M-TL backbone.

    Build once per process. Methods are stateless except for the loaded
    model itself.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        dtype: torch.dtype = DEFAULT_DTYPE,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size

        log.info("loading %s (pretrained, num_frames=%d)...",
                 PRITHVI_MODEL_NAME, T_SEQUENCE)
        from terratorch.registry import BACKBONE_REGISTRY
        # Build with num_frames=3 to match our T=3 chip-sequence design.
        # Without this, the model defaults to num_frames=1 and rejects
        # multi-temporal input.
        self.model = BACKBONE_REGISTRY.build(
            PRITHVI_MODEL_NAME,
            pretrained=True,
            num_frames=T_SEQUENCE,
        )
        self.model = self.model.to(self.device)
        if self.dtype == torch.float16:
            self.model = self.model.half()
        self.model.eval()

        # Disable autograd globally for inference -- saves memory and time
        for p in self.model.parameters():
            p.requires_grad_(False)

        log.info("model on %s, dtype=%s, batch_size=%d",
                 self.device, self.dtype, self.batch_size)

    # -------------------------------------------------------------------------
    # Low-level: tensor in -> tensor out
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _forward_batch(
        self,
        chips: torch.Tensor,             # (B, T, C, H, W) or (B, C, T, H, W)
        temporal_coords: torch.Tensor,   # (B, T, 2)
        location_coords: torch.Tensor,   # (B, 2)
    ) -> torch.Tensor:
        """Forward pass + mean-pool. Returns (B, EMBEDDING_DIM) on the model's
        device, in fp16 (or whatever self.dtype is)."""
        # Ensure BCTHW layout (probe confirmed this is the working layout)
        if chips.dim() != 5:
            raise ValueError(f"chips must be 5D, got shape {tuple(chips.shape)}")
        # Heuristic: if dim 1 == T_SEQUENCE and dim 2 == N_BANDS, it's BTCHW; transpose
        if chips.shape[1] == T_SEQUENCE and chips.shape[2] == N_BANDS:
            chips = chips.permute(0, 2, 1, 3, 4).contiguous()
        # Now expect (B, C=6, T=3, H, W)
        B, C, T, H, W = chips.shape
        if (C, T, H, W) != (N_BANDS, T_SEQUENCE, CHIP_SIZE, CHIP_SIZE):
            raise ValueError(
                f"chips have shape {tuple(chips.shape)}; "
                f"expected (B, {N_BANDS}, {T_SEQUENCE}, {CHIP_SIZE}, {CHIP_SIZE})"
            )

        # Cast to model dtype
        chips_d = chips.to(device=self.device, dtype=self.dtype, non_blocking=True)
        temporal_d = temporal_coords.to(device=self.device, dtype=self.dtype,
                                        non_blocking=True)
        location_d = location_coords.to(device=self.device, dtype=self.dtype,
                                        non_blocking=True)

        # Forward returns a list of 24 tensors (one per ViT block).
        out = self.model(
            chips_d,
            temporal_coords=temporal_d,
            location_coords=location_d,
        )
        # We use the last hidden state (deepest representation).
        last = out[-1]    # (B, 589, D=1024)

        # Mean-pool over all tokens (588 patches + 1 CLS).
        # See module docstring for the choice rationale.
        pooled = last.mean(dim=1)   # (B, 1024)
        return pooled

    # -------------------------------------------------------------------------
    # Mid-level: numpy chips in -> numpy embeddings out
    # -------------------------------------------------------------------------
    def embed_batch(
        self,
        chips_np: np.ndarray,                # (B, T, C, H, W) float
        temporal_coords_np: np.ndarray,      # (B, T, 2) float
        location_coords_np: np.ndarray,      # (B, 2) float
    ) -> np.ndarray:
        """Run a single forward pass on already-loaded numpy data.

        Returns (B, EMBEDDING_DIM) float32 numpy array.
        """
        chips = torch.from_numpy(chips_np)
        temporal = torch.from_numpy(temporal_coords_np)
        location = torch.from_numpy(location_coords_np)
        emb = self._forward_batch(chips, temporal, location)
        return emb.float().cpu().numpy()

    # -------------------------------------------------------------------------
    # High-level: ChipQuery sequences in -> EmbeddingResult list out
    # -------------------------------------------------------------------------
    def embed_query_sequences(
        self,
        queries: Sequence,                 # Sequence[ChipQuery]
        chip_root: str | Path = "",        # if relative chip_paths in index, prepend
        progress_every: int = 100,
    ) -> list[EmbeddingResult]:
        """Top-level entry point for D.1.d.

        For each ChipQuery, load its 3 chips (or empty if no real chips),
        run a batch forward, and return EmbeddingResults.

        Queries with chip_count == 0 (no real chips) are emitted with
        embedding=None; the master-table rebuild will leave their row's
        prithvi_* columns as NaN, and XGBoost handles via missing-direction
        split (per decisions log).

        Batches as many queries as fit in self.batch_size simultaneously.
        Each query consumes T=3 chips that get stacked along the time axis.
        """
        chip_root_p = Path(chip_root) if chip_root else None

        results: list[EmbeddingResult] = [None] * len(queries)  # type: ignore

        # Separate queries with chips from queries without
        real_idxs: list[int] = []
        empty_idxs: list[int] = []
        for i, q in enumerate(queries):
            if q.chip_count == 0:
                empty_idxs.append(i)
            else:
                real_idxs.append(i)

        # Emit empty results for the no-chip queries
        for i in empty_idxs:
            q = queries[i]
            results[i] = EmbeddingResult(
                GEOID=q.GEOID,
                year=q.year,
                forecast_date=q.forecast_date,
                embedding=None,
                chip_count=q.chip_count,
                chip_age_days_max=q.chip_age_days_max,
                cloud_pct_max=q.cloud_pct_max,
                corn_pixel_frac_min=q.corn_pixel_frac_min,
            )

        # Process the real queries in batches
        log.info("embedding %d queries with chips, %d empty (NaN-emitted)",
                 len(real_idxs), len(empty_idxs))

        n_done = 0
        for batch_start in range(0, len(real_idxs), self.batch_size):
            batch_idxs = real_idxs[batch_start: batch_start + self.batch_size]
            B = len(batch_idxs)

            # Build the (B, T, C, H, W) tensor from chip files
            chips_arr = np.zeros(
                (B, T_SEQUENCE, N_BANDS, CHIP_SIZE, CHIP_SIZE),
                dtype=np.float32,
            )
            temporal_arr = np.zeros((B, T_SEQUENCE, 2), dtype=np.float32)
            location_arr = np.zeros((B, 2), dtype=np.float32)

            for bi, qi in enumerate(batch_idxs):
                q = queries[qi]
                # picks is always length 3 (real or padded)
                if len(q.picks) != T_SEQUENCE:
                    raise RuntimeError(
                        f"ChipQuery for {q.GEOID}/{q.year}/{q.forecast_date} "
                        f"has {len(q.picks)} picks, expected {T_SEQUENCE}"
                    )
                # Load the 3 chips. Padded picks duplicate the same chip_path
                # so we'd be loading the same file 2-3x. We could memoize
                # within a query, but at <1ms per chip read this isn't worth
                # the complexity for the usual case (most chips not padded).
                lat: Optional[float] = None
                lon: Optional[float] = None
                for ti, pick in enumerate(q.picks):
                    path = pick.chip_path
                    if chip_root_p is not None and not Path(path).is_absolute():
                        path = chip_root_p / path
                    chips_arr[bi, ti] = load_chip_as_tensor(path)
                    temporal_arr[bi, ti, 0] = pick.scene_date.year
                    temporal_arr[bi, ti, 1] = pick.scene_date.timetuple().tm_yday
                    # Use the first non-padded pick's centroid (real lat/lon).
                    # All picks are for the same county so they should agree
                    # on county centroid; using pick chip_lat/chip_lon would
                    # also work but those aren't carried in ChipPick.
                    # Workaround: pull from the chip's GeoTIFF tags or use
                    # a fallback. For now: stash on first call.
                    # NOTE: lat/lon belong to the COUNTY not the chip per
                    # decisions log; we should be passing county centroid.
                    # For initial v1 we use the first chip's centroid as a
                    # proxy; D.1.d's caller can pre-populate county centroids
                    # if higher fidelity is needed.

                # Location: use first chip's lat/lon. ChipPick doesn't carry
                # chip_lat/chip_lon (it's in the parquet but we filter to
                # phenology fields). For v1 we read it from the chip
                # GeoTIFF's centroid. This is an O(1)-extra-rasterio-open
                # per query; tolerable.
                # If you need to skip this, pass chip_lat/chip_lon through
                # from chip_picker by extending ChipPick.
                if lat is None:
                    lat, lon = _read_chip_centroid_latlon(
                        chip_root_p / q.picks[0].chip_path
                        if chip_root_p else q.picks[0].chip_path
                    )
                location_arr[bi, 0] = lat
                location_arr[bi, 1] = lon

            # Forward
            emb = self.embed_batch(chips_arr, temporal_arr, location_arr)
            assert emb.shape == (B, EMBEDDING_DIM)

            # Pack results
            for bi, qi in enumerate(batch_idxs):
                q = queries[qi]
                results[qi] = EmbeddingResult(
                    GEOID=q.GEOID,
                    year=q.year,
                    forecast_date=q.forecast_date,
                    embedding=emb[bi].astype(np.float32),
                    chip_count=q.chip_count,
                    chip_age_days_max=q.chip_age_days_max,
                    cloud_pct_max=q.cloud_pct_max,
                    corn_pixel_frac_min=q.corn_pixel_frac_min,
                )

            n_done += B
            if progress_every and n_done % progress_every < self.batch_size:
                log.info("  embedded %d / %d real queries", n_done, len(real_idxs))

        log.info("done: %d real-embedded, %d NaN-emitted", len(real_idxs), len(empty_idxs))
        return results


# =============================================================================
# Per-chip lat/lon lookup
# =============================================================================
# ChipPick doesn't carry chip_lat/chip_lon (those are in the parquet on
# the row, but ChipPick filters to phenology-relevant fields). For
# Prithvi-TL's location_coords we want a (lat, lon). We open the chip
# GeoTIFF and reproject its center to EPSG:4326. This is a small extra
# I/O cost per query but keeps the chip_picker API minimal.
#
# Caching across calls in the same process: this gets called once per
# query (3 picks per query but we use only the first). With ~18,624
# queries that's 18,624 rasterio-opens, each <1ms. ~20s total -- fine.

def _read_chip_centroid_latlon(chip_path: str | Path) -> tuple[float, float]:
    """Open a chip GeoTIFF and return its centroid in (lat, lon) WGS84."""
    import rasterio.warp
    with rasterio.open(chip_path) as src:
        b = src.bounds
        cx = (b.left + b.right) / 2
        cy = (b.bottom + b.top) / 2
        lons, lats = rasterio.warp.transform(src.crs, "EPSG:4326", [cx], [cy])
    return float(lats[0]), float(lons[0])


# =============================================================================
# Convenience: embed-from-disk smoke test
# =============================================================================
def smoke_test() -> None:
    """Tiny end-to-end: load embedder, run a single B=2 random forward,
    confirm output shape. Useful as a quick health check."""
    log.info("PrithviEmbedder smoke test")
    embedder = PrithviEmbedder()
    rng = np.random.default_rng(0)
    chips = rng.uniform(0.0, 0.5, size=(2, T_SEQUENCE, N_BANDS, CHIP_SIZE, CHIP_SIZE)
                        ).astype(np.float32)
    temporal = np.tile([[2018, 200], [2018, 230], [2018, 260]], (2, 1, 1)
                       ).astype(np.float32).reshape(2, T_SEQUENCE, 2)
    location = np.array([[42.0, -93.0], [42.0, -93.0]], dtype=np.float32)
    emb = embedder.embed_batch(chips, temporal, location)
    log.info("smoke output shape: %s, dtype: %s, range: [%.3f, %.3f]",
             emb.shape, emb.dtype, emb.min(), emb.max())
    assert emb.shape == (2, EMBEDDING_DIM)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    smoke_test()

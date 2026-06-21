# Location Intelligence

This is the layer that makes the system **coordinate-first**: it turns a raw latitude/longitude into everything the forecasting model needs to know about that location's history.

## Coordinate-Aware Feature Store

File: `src/inference/feature_store.py` (`build_location_and_cluster_store`, `build_feature_store`)

At prediction time, the user only provides:

```text
latitude
longitude
event cause
vehicle type
road closure
crowd size
weather
```

But the forecasting model needs:

```text
lag features
rolling features
corridor average
corridor volatility
zone risk
junction risk
cause risk
closure risk
cluster risk
calendar-aware event context
```

The feature store bridges this gap by holding precomputed historical values, keyed by location and time, so they can be looked up instantly instead of recomputed from raw history on every request.

Stored objects include:

```text
corridor_hour_profiles
corridor_profiles
global_profile
corridor_location_profiles
corridor_location_points
hotspot_points
spatial_cluster_model
spatial_cluster_centers
spatial_cluster_hour_profiles
spatial_cluster_profiles
risk thresholds
```

This store is built as part of the full training pipeline (`train_all.py`) and can also be built independently via `scripts/prepare_feature_store.py`.

## Location Resolver

File: `src/inference/location_resolver.py`

Takes `latitude`, `longitude`, and the feature store, and returns location intelligence such as:

```python
{
    "corridor": "ORR East 1",
    "matched_by": "nearest real corridor historical point",
    "distance_m": 420.5,
    "confidence": "HIGH",
    "spatial_cluster_id": 7,
    "nearest_hotspot_distance_m": 180.4,
    "spatial_density_at_point": 0.72
}
```

It performs:

- coordinate validation
- Bengaluru boundary validation
- restricted-zone validation (`src/inference/location_validity_guard.py`)
- nearest real corridor matching
- nearest corridor centroid matching
- nearest spatial cluster lookup
- nearest hotspot distance calculation
- spatial density calculation

## Bengaluru Bounding Box Validation

The system validates that submitted coordinates fall inside Bengaluru's coverage area. For example, a coordinate of `28.6139, 77.2090` is Delhi, not Bengaluru — the system rejects this rather than attempting to match it to a corridor:

```text
Selected location is outside Bengaluru coverage area.
Please choose a location within Bengaluru.
```

This prevents fake or meaningless corridor matching for out-of-city coordinates.

## Restricted-Zone Validation

File: `src/inference/location_validity_guard.py`

In addition to the city-wide boundary check above, coordinates are checked against a separate list of restricted zones (loaded from `data/restricted_zones.csv` when present). This lets specific no-go or sensitive areas be flagged distinctly from a simple "outside Bengaluru" rejection, without requiring those areas to be hard-coded into the boundary check itself.

## Location Resolution Tiers

The location resolver returns one of six match states, in order of specificity, using the first one available:

| Order | `matched_by` value | Meaning |
|---|---|---|
| 1 | `nearest real corridor historical point` | A real historical event point is close to the input coordinates |
| 2 | `nearest real corridor centroid` | No close point, but a known corridor's centroid is close |
| 3 | `nearest real corridor inside spatial cluster` | No close corridor, but the surrounding spatial cluster has one |
| 4 | `spatial cluster fallback; no reliable corridor label` | Nothing corridor-specific is close enough; cluster-level profile is used |
| — | `outside Bengaluru boundary` | Coordinates fall outside the validated coverage area — rejected, not matched |
| — | `invalid coordinates fallback` | Coordinates are malformed or unparseable — rejected, not matched |

Each successful tier (1–4) carries a `confidence` value (HIGH / MEDIUM / LOW) that the dashboard surfaces directly in the "Coordinate Resolver" panel, so a weak match is never presented with false certainty.

### Why It's Needed

If a coordinate can't be matched confidently to a known corridor, the model should not fall back to all-zero lag features.

**Old (weak) behavior:**

```text
unknown corridor
        ↓
lag_1 = 0
rolling_24 = 0
corridor_avg = 0
        ↓
weak prediction
```

**New behavior:**

```text
unknown or weak location
        ↓
find nearest KMeans spatial cluster
        ↓
use cluster-hour historical profile
        ↓
prediction still has meaningful history
```

## Cluster Fallback Ablation Study

File: `src/evaluation/cluster_fallback_ablation.py`

**Purpose:** test whether the cluster fallback is actually useful, rather than just asserting that it is.

The ablation compares:

- **Method A:** normal corridor-hour profiles
- **Method B:** forced spatial cluster fallback profiles

Recent result:

```text
Rows tested       : 5000
Normal MAE        : 0.1537
Cluster MAE       : 0.2384
MAE Delta         : -0.0848
Improvement       : -55.18%
```

**Conclusion:** cluster fallback is weaker than corridor-hour history when corridor matching is reliable. It is therefore **not** used as a replacement for corridor history — it is used only when corridor matching is weak or unavailable (tier 4 above). This makes the fallback design evidence-based rather than assumed.

Output: `models/cluster_fallback_ablation.json`

## Related Docs

- [Data & Features](data-and-features.md) — what the feature store actually stores
- [Forecasting Model](forecasting-model.md) — how resolved features feed the model, and how confidence level determines which tier is used
- [Judge / Reviewer Notes](judge-notes.md) — how to explain the fallback honestly

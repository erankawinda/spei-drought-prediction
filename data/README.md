# Data record

## Files in this repository

| Station label | Rows | Period | SHA-256 |
|---|---:|---|---|
| Buttala | 1,428 | 1901-01 to 2019-12 | `b2646f772f521e270d6f47e593ac70653d56548fcfca932058609b153b4f0061` |
| Padaviya | 1,428 | 1901-01 to 2019-12 | `f82b2efb1c9d56a73b0b87f59adbcc1f4801810c0db96bee35f46280e539c337` |
| Tissamaharama | 1,428 | 1901-01 to 2019-12 | `dabfe80857c0bd35da7f83f6a49b3a2b6295a6c2392aa4b33ebb0d347116a254` |

Each file has one unique, uninterrupted record per calendar month. The hashes
are frozen in both `SHA256SUMS` and `configs/benchmark_v1.toml`; the loader stops
before analysis if any byte changes.

## Reconstructed provenance

The surviving 2021 paper, abstract, presentation, and reviewer response say that
the meteorological series came from the Climatic Research Unit (CRU), University
of East Anglia, via the CEDA archive; cover January 1901 through December 2019;
and have 0.5 x 0.5 degree spatial resolution. The presentation states that SPEI
was calculated from rainfall and minimum/maximum temperature using the
Hargreaves PET method because the longer records required for Penman-Monteith
were unavailable.

These details align with the official [CRU TS v4.04 archive](https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.04/),
whose period is 1901-2019 and whose variable set matches the recovered tables.
The historical exports themselves do **not** retain the CRU version string,
grid-cell coordinates, original filenames, or a download manifest. Version
4.04 is therefore the best-supported reconstruction, not a claim that can be
verified byte-for-byte against the upstream grid.

## Fields and units

The upstream units below follow the official
[CRU variable definitions](https://crudata.uea.ac.uk/cru/data/hrg/index.htm).
Names in the recovered files are retained even where they are imperfect.

| Recovered field | Interpretation | Unit / status | Used by benchmark |
|---|---|---|---|
| `cloud_cover` | cloud cover | % | yes |
| `mean_prep` | monthly precipitation (`pre`) | mm/month | yes |
| `pot_evap` | CRU potential evapotranspiration (`pet`) | mm/day | yes |
| `mean_tmp` | monthly average daily mean temperature | degrees C | yes |
| `max_tmp`, `min_tmp` | monthly average daily maximum/minimum temperature | degrees C | no; redundant with mean/range |
| `diurnal_tmp_range` | diurnal temperature range | degrees C | yes |
| `vapour_pressure` | vapour pressure | hPa | yes |
| `wet_day_frq` | wet-day frequency | days/month | yes |
| `PET` / canonical `pet` | locally derived PET total used during SPEI preparation | likely mm/month; exact implementation unrecorded | no |
| `spei1`, `spei3`, `spei6`, `spei9`, `spei12` | SPEI at the named accumulation window | dimensionless | selected target only |

The analysis never subtracts `pot_evap` from precipitation and excludes the
locally derived `pet` field because its exact derivation is not preserved.

## Validation and missingness

The loader uses an explicit header allow-list, normalizes historical SPEI header
spacing, and rejects unknown or colliding columns. It verifies calendar
continuity, finite meteorological predictors, temperature ordering, physical
ranges, and the expected leading SPEI warm-up gaps:

| Target | Leading missing months per station |
|---|---:|
| SPEI-1 | 0 |
| SPEI-3 | 2 |
| SPEI-6 | 5 |
| SPEI-9 | 8 |
| SPEI-12 | 11 |

No observation is forward-filled, backward-filled, interpolated, winsorized, or
deleted as an outlier. A warm-up value is removed only when a target-specific
sample cannot be formed from it.

## Scope and reuse

The recovered data cover three labels only. Ratnapura and the IOD/ENSO inputs
described in the 2021 work are absent. The SPEI distribution, fitting procedure,
and calibration period are also not recorded, so target-construction leakage
cannot be ruled out even though the forecasting pipeline itself is temporally
controlled.

CRU TS v4.04 is offered on its version-specific page under the Open Government
Licence with attribution to the Climatic Research Unit (University of East
Anglia) and the Met Office. The data files in this directory remain subject to
those upstream terms and are not covered by this repository's MIT code licence.

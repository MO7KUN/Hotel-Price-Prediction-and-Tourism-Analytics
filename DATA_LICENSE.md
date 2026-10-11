# Data, models and research-output license

## Attribution

Silvia & Óscar (2025). **Alojamientos y precios en Booking.com (Barcelona, 2026)**, version 1. Zenodo. DOI: https://doi.org/10.5281/zenodo.17560799. Original source platform: Booking.com.

The Zenodo API identifies the dataset license as **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**.

License terms: https://creativecommons.org/licenses/by-nc-sa/4.0/
Legal code: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en

## Derivative outputs and modifications

The processed data, fitted research model artifacts, figures, notebook narrative, dashboard data and analytical documentation in this project are distributed under CC BY-NC-SA 4.0 with the attribution above. No endorsement by the original authors or Booking.com is implied.

Modifications performed on 2026-10-11: canonical URL extraction and tracking-parameter removal; currency and distance normalization; occupancy validation; exclusion of 75 contradictory room-count rows and 150 multi-room quotes; extraction of text-keyword features; property-disjoint model training; aggregation, prediction and three-cluster descriptive segmentation. No synthetic hotel observations were introduced.

You must retain attribution, indicate changes, use these data and derived outputs for noncommercial purposes and share adapted material under the same license. The data license is separate from the source-code license.

## Source code

Original Python implementation, dashboard interface code and tests are MIT licensed, as stated in `LICENSE`. Source data are not relicensed as MIT. A commercial use of the research dataset or derived outputs requires separate permission or a different suitably licensed dataset.

## Provenance

`data/source_manifest.json` records the versioned original URL, DOI, authors, download timestamp, original MD5, exact SHA256 and byte count. Raw CSV and cached HTML are excluded from Git. The committed processed cohort contains sanitized public property URLs with no session or tracking query strings.

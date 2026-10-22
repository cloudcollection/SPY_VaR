# SPY data snapshot

The canonical input for new experiments is `code/data/spy_data.csv`.
The file is read without modification. Chapter-local copies are historical
artifacts and are not read by the new pipeline.

## Contents verified from the snapshot

- Instrument: SPY, as identified by the original study.
- Rows: 4,640; dates: 2000-01-04 through 2018-06-27.
- First CSV column: saved date index (blank header).
- `log_ret`: daily log return, in decimal units (0.01 means approximately 1%).
- `rv5`: described in the original report as a realized-volatility measure.
- `bv`: described in the original report as a bipower-variation measure.
- All three numeric columns are finite in the checked-in snapshot.

## Integrity and timing

Each new run records the SHA-256 of the complete input CSV and the actual prefix
or full sample used. Loaders reject invalid/duplicate/out-of-order dates and
missing/nonfinite numeric values. They do not silently impute, reorder or drop
observations. Features and volatility forecasts use information preceding their
target return; see the reproduction guide for the C2 alignment.

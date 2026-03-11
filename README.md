# kubera

An Indian stock movement prediction system that forecasts next-day stock direction using two approaches.

## Local bootstrap

```powershell
$env:PYTHONPATH='src'
python -m kubera.bootstrap
python -m kubera.ingest.market_data
python -m pytest
```

Provider keys are not required for the local bootstrap commands above.

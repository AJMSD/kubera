from kubera.config import load_settings
from kubera.pilot.live_pilot import run_live_pilot
import pandas as pd
import datetime
from unittest.mock import patch

def main():
    settings = load_settings()    
    print("Running market prediction for March 16th... (pre_market)")
    
    # Target time: March 16, 2026, 08:05:00 IST = 02:35:00 UTC
    target_utc = datetime.datetime(2026, 3, 16, 2, 35, 0, tzinfo=datetime.timezone.utc)
    
    class MockDatetime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            if tz == datetime.timezone.utc:
                return target_utc
            return target_utc.astimezone(tz)
    
    with patch("datetime.datetime", MockDatetime):
        with patch("kubera.pilot.live_pilot.datetime", MockDatetime):
            with patch("kubera.utils.time_utils.datetime", MockDatetime):
                result = run_live_pilot(
                    settings=settings,
                    prediction_mode="pre_market",
                    explain=True
                )
    print("Prediction complete.")

if __name__ == "__main__":
    main()

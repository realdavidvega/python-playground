from alpha_vantage.timeseries import TimeSeries
from pydantic import BaseModel, ConfigDict


class AlphaVantageResources(BaseModel):
    time_series: TimeSeries
    model_config = ConfigDict(arbitrary_types_allowed=True)

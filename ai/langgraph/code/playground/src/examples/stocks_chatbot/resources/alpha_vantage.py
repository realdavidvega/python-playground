from alpha_vantage.timeseries import TimeSeries
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, frozen=True))
class AlphaVantageResources:
    time_series: TimeSeries

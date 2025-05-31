from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.timeseries import TimeSeries
from fredapi import Fred
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, frozen=True))
class FinanceResources:
    time_series: TimeSeries
    foreign_exchange: ForeignExchange
    fred: Fred

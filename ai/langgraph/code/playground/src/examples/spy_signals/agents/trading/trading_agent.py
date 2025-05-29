import pandas as pd
import talib
from langchain_core.language_models import LanguageModelLike
from langchain_core.tools import tool
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent


def build_trading_agent(model: LanguageModelLike, debug: bool = False) -> CompiledGraph:
    @tool
    def generate_trading_signal(
        spy_daily: float,
        usd_eur_rate: float,
        interest_rates: list[float],
    ) -> str:
        """
        Generates SPY trading signals using daily price data, USD/EUR rate, and current interest rate.
        Returns 'buy', 'sell', or 'hold'.
        """

        # Create DataFrame with required features
        df = pd.DataFrame(
            [
                {
                    "spy": spy_daily,
                    "usd_eur": usd_eur_rate,
                    "interest_rates": interest_rates,
                }
            ]
        )

        # Technical Indicators
        df["RSI"] = talib.RSI(df["spy"], timeperiod=14)
        df["MA50"] = df["spy"].rolling(50).mean()
        df["MA200"] = df["spy"].rolling(200).mean()
        macd, macd_signal, _ = talib.MACD(df["spy"])
        df["MACD_Hist"] = macd - macd_signal

        # Macro Trends (5-day window)
        latest = df.iloc[-1]
        rate_trend = (latest["interest_rates"] - 3) / df["interest_rate"].iloc[4]
        usd_eur_trend = (latest["usd_eur"] - df["usd_eur"].iloc[-5]) / df[
            "usd_eur"
        ].iloc[-5]

        # Signal Conditions
        buy_conditions = [
            latest["RSI"] < 35,
            latest["MA50"] > latest["MA200"],
            latest["MACD_Hist"] > 0,
            rate_trend < -0.02,  # 2% rate decrease
            usd_eur_trend < -0.01,  # USD weakening vs EUR
        ]

        sell_conditions = [
            latest["RSI"] > 65,
            latest["MA50"] < latest["MA200"],
            latest["MACD_Hist"] < 0,
            rate_trend > 0.02,  # 2% rate increase
            usd_eur_trend > 0.01,  # USD strengthening
        ]

        if sum(buy_conditions) >= 4:
            return "buy"
        elif sum(sell_conditions) >= 4:
            return "sell"
        return "hold"

    return create_react_agent(
        model=model,
        tools=[generate_trading_signal],
        prompt=(
            """
            You are a trading agent that generates trading signals based on daily price data, USD/EUR rate, and current interest rate.
            INSTRUCTIONS:
            - Assist ONLY with trading signals, do not perform calculations or use technical analysis
            - Be very detailed, even verbose, and not ambiguous in your queries
            - After you're done with your tasks, respond to the supervisor directly
            - Respond ONLY with the results of your work, do NOT include ANY other text
            """
        ),
        name="trading_agent",
        debug=debug,
    )

import pandas as pd
import talib
from langchain_core.language_models import LanguageModelLike
from langchain_core.tools import tool
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent


def build_trading_agent(model: LanguageModelLike, debug: bool = False) -> CompiledGraph:
    @tool
    def generate_trading_signal(
        spy_daily: list[float],
        usd_eur_rates: list[float],
        interest_rates: list[float],
    ) -> str:
        """
        Generates SPY trading signals using daily price data, USD/EUR rate, and current interest rate.
        Each list contains the last 20 entries of data.
        Returns 'buy', 'sell', or 'hold'.
        """

        print(
            f"Called generate_trading_signal tool: {spy_daily}, {usd_eur_rates}, {interest_rates}"
        )

        df = pd.DataFrame(
            {
                "spy": spy_daily,
                "usd_eur_rates": usd_eur_rates,
                "interest_rates": interest_rates,
            }
        )

        df["RSI"] = talib.RSI(df["spy"], timeperiod=14)
        df["MA10"] = df["spy"].rolling(10).mean()
        df["MA20"] = df["spy"].rolling(20).mean()
        macd, macd_signal, _ = talib.MACD(df["spy"])
        df["MACD_Hist"] = macd - macd_signal

        # Macro Trends (5-day window)
        df["usd_trend"] = df["usd_eur_rates"].pct_change(5)
        df["rate_trend"] = df["interest_rates"].pct_change(5)

        # Latest values
        latest = df.iloc[-1]

        # Signal Conditions
        buy_conditions = [
            latest["RSI"] < 45,
            latest["MA10"] > latest["MA20"],
            latest["MACD_Hist"] > 0,
            latest["rate_trend"] < -0.02,  # 2% rate decrease
            latest["usd_trend"] < -0.01,  # USD weakening vs EUR
        ]

        sell_conditions = [
            latest["RSI"] > 65,
            latest["MA10"] < latest["MA20"],
            latest["MACD_Hist"] < 0,
            latest["rate_trend"] > 0.02,  # 2% rate increase
            latest["usd_trend"] > 0.01,  # USD strengthening
        ]

        if sum(buy_conditions) >= 3:
            return "buy"
        elif sum(sell_conditions) >= 3:
            return "sell"
        return "hold"

    return create_react_agent(
        model=model,
        tools=[generate_trading_signal],
        prompt=(
            """
            You are a trading agent that generates trading signals based on daily price data, USD/EUR rates, and current interest rates.
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

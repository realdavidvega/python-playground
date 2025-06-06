from dataclasses import field

from langchain_core.language_models import LanguageModelLike
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


def _default_model():
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, frozen=True))
class AgentsConfig:
    debug: bool = False
    interest_agent_model: LanguageModelLike = field(default_factory=_default_model)
    usd_agent_model: LanguageModelLike = field(default_factory=_default_model)
    spy_agent_model: LanguageModelLike = field(default_factory=_default_model)
    supervisor_model: LanguageModelLike = field(default_factory=_default_model)
    trading_agent_model: LanguageModelLike = field(default_factory=_default_model)

import os
import dotenv
import random
import jsonlines
from langchain_openai import ChatOpenAI
from prompt import (
    DATA_TRANSLATION_PROMPT,
    FLUENCY_PROMPT,
    ACCURACY_PROMPT,
    NATURALNESS_PROMPT,
    CS_RATIO_PROMPT,
    SOCIAL_CULTURAL_PROMPT,
    REFINER_PROMPT,
)
from node_models import (
    AgentRunningState,
    TranslationResponse,
    AccuracyResponse,
    FluencyResponse,
    NaturalnessResponse,
    CSRatioResponse,
    SocialCulturalResponse,
)
from utils import weighting_scheme
from copy import deepcopy

from typing import Dict, Any


dotenv.load_dotenv()

API_KEY = os.getenv("API_KEY")
MODEL = "gpt-4o"
OUTPUT_DIR = "data_output"

##add in function for translating the xnli dataset


def RunDataTranslationAgent(state: AgentRunningState):
    DataTranslationAgent = DATA_TRANSLATION_PROMPT | ChatOpenAI(
        model=MODEL, temperature=0.7, api_key=API_KEY
    ).with_structured_output(TranslationResponse)
    response = DataTranslationAgent.invoke(state)
    retry = 4
    if not response.get("hypo"):
        while retry > 0:
            response = DataTranslationAgent.invoke(state)
            if response.get("hypo"):
                break
            retry -= 1
    return {"data_generation_result": response["hypo"]}

def RunAccuracyAgent(state: AgentRunningState):
    AccuracyAgent = ACCURACY_PROMPT | ChatOpenAI(
        model=MODEL, temperature=0.1, api_key=API_KEY
    ).with_structured_output(AccuracyResponse)
    response = AccuracyAgent.invoke(state)

    return {"accuracy_result": response}


def RunFluencyAgent(state: AgentRunningState):
    FluencyAgent = FLUENCY_PROMPT | ChatOpenAI(
        model=MODEL, temperature=0.1, api_key=API_KEY
    ).with_structured_output(FluencyResponse)
    response = FluencyAgent.invoke(state)

    return {"fluency_result": response}


def RunNaturalnessAgent(state: AgentRunningState):
    NaturalnessAgent = NATURALNESS_PROMPT | ChatOpenAI(
        model=MODEL, temperature=0.1, api_key=API_KEY
    ).with_structured_output(NaturalnessResponse)
    response = NaturalnessAgent.invoke(state)

    return {"naturalness_result": response}


def RunCSRatioAgent(state: AgentRunningState):
    CSRatioAgent = CS_RATIO_PROMPT | ChatOpenAI(
        model=MODEL, temperature=0.1, api_key=API_KEY
    ).with_structured_output(CSRatioResponse)
    response = CSRatioAgent.invoke(state)

    return {"cs_ratio_result": response}


def RunSocialCulturalAgent(state: AgentRunningState):
    SocialCulturalAgent = SOCIAL_CULTURAL_PROMPT | ChatOpenAI(
        model=MODEL, temperature=0.1, api_key=API_KEY
    ).with_structured_output(SocialCulturalResponse)
    response = SocialCulturalAgent.invoke(state)
    return {"social_cultural_result": response}


def SummarizeResult(state: AgentRunningState):
    summary = f"""
    data_generation_result: {state["data_generation_result"]}
    Fluency Result: {state["fluency_result"]}
    Naturalness Result: {state["naturalness_result"]}
    CSRatio Result: {state["cs_ratio_result"]}
    """
    state["summary"] = summary
    # print(summary)
    # with jsonlines.open("result/summary_result_new.jsonl", "a") as f:
    #     f.write(state)

    return {"score": weighting_scheme(state), "summary": summary}


def AcceptanceAgent(state: AgentRunningState):
    language = state["first_language"]
    with jsonlines.open(
        f"{OUTPUT_DIR}/{language}.jsonl",
        "a",
    ) as f:
        f.write(state)
    return


def RunRefinerAgent(state: AgentRunningState):

    RefinerAgent = REFINER_PROMPT | ChatOpenAI(
        model=MODEL, temperature=0.1, api_key=API_KEY
    ).with_structured_output(TranslationResponse)
    response = RefinerAgent.invoke(state)

    return {"refiner_result": response, "refine_count": 3}

def RunMCPAgent(state: AgentRunningState) -> Dict[str, Any]:
    """
    Iterate through all MCP tools in the registry, execute them in order, and merge the results.
    The execution result -> state["mcp_result"], used by the subsequent nodes.
    """
    result: Dict[str, Any] = {}
    for tool_name, tool in get_all_tools().items():
        try:
            result.update(tool.run(state))
        except Exception as e:
            # Ensure that a tool failure does not affect the subsequent nodes
            result[tool_name] = f"ERROR: {e}"
    return {"mcp_result": result}
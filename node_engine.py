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
    # CS_RATIO_PROMPT,
    #SOCIAL_CULTURAL_PROMPT,
    REFINER_PROMPT,
)
from node_models import (
    AgentRunningState,
    TranslationResponse,
    AccuracyResponse,
    FluencyResponse,
    NaturalnessResponse,
    # CSRatioResponse,
    #SocialCulturalResponse,
)
from read_xnli_dataset import XNLIDataLoader
from utils import weighting_scheme,save_jsonl_to_tsv, get_premise_label
from copy import deepcopy

from typing import Dict, Any


dotenv.load_dotenv()

API_KEY = os.getenv("API_KEY")
MODEL = os.getenv("MODEL")
OUTPUT_DIR = "data_output"
TEMPERATURE=1

##add in function for translating the xnli dataset

loader = XNLIDataLoader(lang='en', test_path='xnli.test.tsv')

def RunDataTranslationAgent(state: AgentRunningState):
    DataTranslationAgent = DATA_TRANSLATION_PROMPT | ChatOpenAI(
        model=MODEL, temperature=TEMPERATURE, api_key=API_KEY
    ).with_structured_output(TranslationResponse)
    response = DataTranslationAgent.invoke(state)
    # retry = 4
    # if not response.get():
    #     while retry > 0:
    #         response = DataTranslationAgent.invoke(state)
    #         if response.get():
    #             break
    #         retry -= 1
    print(response)
    state["data_translation_result"] = response
    return {"data_translation_result": response}

def RunAccuracyAgent(state: AgentRunningState):
    AccuracyAgent = ACCURACY_PROMPT | ChatOpenAI(
        model=MODEL, temperature=TEMPERATURE, api_key=API_KEY
    ).with_structured_output(AccuracyResponse)
    response = AccuracyAgent.invoke(state)
    print(response)
    return {"accuracy_result": response}


def RunFluencyAgent(state: AgentRunningState):
    FluencyAgent = FLUENCY_PROMPT | ChatOpenAI(
        model=MODEL, temperature=TEMPERATURE, api_key=API_KEY
    ).with_structured_output(FluencyResponse)
    response = FluencyAgent.invoke(state)
    print(response)
    return {"fluency_result": response}


def RunNaturalnessAgent(state: AgentRunningState):
    NaturalnessAgent = NATURALNESS_PROMPT | ChatOpenAI(
        model=MODEL, temperature=TEMPERATURE, api_key=API_KEY
    ).with_structured_output(NaturalnessResponse)
    response = NaturalnessAgent.invoke(state)
    print(response)
    return {"naturalness_result": response}


# def RunCSRatioAgent(state: AgentRunningState):
#     CSRatioAgent = CS_RATIO_PROMPT | ChatOpenAI(
#         model=MODEL, temperature=TEMPERATURE, api_key=API_KEY
#     ).with_structured_output(CSRatioResponse)
#     response = CSRatioAgent.invoke(state)
#     print(response)
#     return {"cs_ratio_result": response}


#def RunSocialCulturalAgent(state: AgentRunningState):
    SocialCulturalAgent = SOCIAL_CULTURAL_PROMPT | ChatOpenAI(
        model=MODEL, temperature=TEMPERATURE, api_key=API_KEY
    ).with_structured_output(SocialCulturalResponse)
    response = SocialCulturalAgent.invoke(state)
    print(response)
    return {"social_cultural_result": response}


def SummarizeResult(state: AgentRunningState):
    summary = f"""
    data_translation_result: {state["data_translation_result"]}
    Accuracy Result: {state["accuracy_result"]}
    Fluency Result: {state["fluency_result"]}
    Naturalness Result: {state["naturalness_result"]}
    """
    state["summary"] = summary
    # print(summary)
    # with jsonlines.open("result/summary_result_new.jsonl", "a") as f:
    #     f.write(state)

    return {"score": weighting_scheme(state), "summary": summary}

def AcceptanceAgent(state: AgentRunningState):
    language = state["second_language"]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    jsonl_file = f"{OUTPUT_DIR}/{language}.jsonl"
    json_dataset_file = f"{OUTPUT_DIR}/{language}_dataset.json"
    tsv_file = f"{OUTPUT_DIR}/cs_{language}_test.tsv"

    # Extract hypothesis text (STRING)
    hypo_text = state.get("hypothesis", {}).get("hypo", "")
    translated_sentence = state["data_translation_result"]["translated_sentence"]

    if os.path.exists(json_dataset_file):
        with jsonlines.open(json_dataset_file, "r") as f:
            existing_translations = list(f)
    with jsonlines.open(jsonl_file, "a") as f:
        f.write(state)

    with jsonlines.open(json_dataset_file, "a") as f:
        f.write(translated_sentence)

    # ---------- Regenerate TSV ----------
    save_jsonl_to_tsv(jsonl_file, tsv_file, loader)



def RunRefinerAgent(state: AgentRunningState):

    RefinerAgent = REFINER_PROMPT | ChatOpenAI(
        model=MODEL, temperature=TEMPERATURE, api_key=API_KEY
    ).with_structured_output(TranslationResponse)
    response = RefinerAgent.invoke(state)
    state["data_translation_result"] = response
    print(f'refiner agent called: {response}')
    return {"refiner_result": response, "refine_count": 1}

# def RunMCPAgent(state: AgentRunningState) -> Dict[str, Any]:
#     """
#     Iterate through all MCP tools in the registry, execute them in order, and merge the results.
#     The execution result -> state["mcp_result"], used by the subsequent nodes.
#     """
#     result: Dict[str, Any] = {}
#     for tool_name, tool in get_all_tools().items():
#         try:
#             result.update(tool.run(state))
#         except Exception as e:
#             # Ensure that a tool failure does not affect the subsequent nodes
#             result[tool_name] = f"ERROR: {e}"
#     return {"mcp_result": result}
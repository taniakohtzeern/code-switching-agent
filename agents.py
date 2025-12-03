import asyncio
from langgraph.graph import StateGraph, START, END
from loguru import logger
from utils import load_config, generate_hypo_list
from node_engine import (
    RunDataTranslationAgent,
    RunAccuracyAgent,
    SummarizeResult,
    RunFluencyAgent,
    RunNaturalnessAgent,
    # RunCSRatioAgent,
    #RunSocialCulturalAgent,
    RunRefinerAgent,
    AcceptanceAgent,
)
from node_models import AgentRunningState
import random
from tqdm import tqdm
import jsonlines as jsl
from datetime import datetime


#agents are adapted from switchlingua
code_switch_lang='vi'
logger.add(f"logs/code_switching_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
config: dict = load_config(f"./config/config_{code_switch_lang}.yaml")
MAX_REFINER_ITERATIONS = 1
start=1200
end=1240


def meet_criteria(state: AgentRunningState):
    if state["score"] < 8 and state["refine_count"] < MAX_REFINER_ITERATIONS:
        return "RefinerAgent"
    else:
        return "AcceptanceAgent"


class CodeSwitchingAgent:
    def __init__(self, scenario_k):
        self.state = AgentRunningState()
        self.state["refine_count"] = 0
        for key in scenario_k.keys():
            self.state[key] = scenario_k[key]
        self.workflow_with_data_generation: StateGraph = (
            self._construct_graph_with_data_generation()
        )

    
    def _construct_graph_with_data_generation(self) -> StateGraph:
        workflow = StateGraph(AgentRunningState)
        workflow.add_node("DataTranslationAgent", RunDataTranslationAgent)
        workflow.add_node("TranslationAdequacyAgent", RunAccuracyAgent)
        workflow.add_node("FluencyAgent", RunFluencyAgent)
        workflow.add_node("NaturalnessAgent", RunNaturalnessAgent)
        # workflow.add_node("CSRatioAgent", RunCSRatioAgent)
        #workflow.add_node("SocialCulturalAgent", RunSocialCulturalAgent)
        workflow.add_node("SummarizeResult", SummarizeResult)
        workflow.add_node("RefinerAgent", RunRefinerAgent)
        workflow.add_node("AcceptanceAgent", AcceptanceAgent)
        # workflow.add_node("NewsGenerationAgent", RunUseToolsAgent)
        workflow.add_edge(START, "DataTranslationAgent")
        # workflow.add_edge(START, "NewsGenerationAgent")
        workflow.add_edge("DataTranslationAgent", "TranslationAdequacyAgent")
        workflow.add_edge("DataTranslationAgent", "FluencyAgent")
        workflow.add_edge("DataTranslationAgent", "NaturalnessAgent")
        # workflow.add_edge("DataTranslationAgent", "CSRatioAgent")
        #workflow.add_edge("DataTranslationAgent", "SocialCulturalAgent")
        workflow.add_edge(
            ["TranslationAdequacyAgent","FluencyAgent", "NaturalnessAgent"],
            "SummarizeResult",
        )
        workflow.add_conditional_edges("SummarizeResult", meet_criteria)
        workflow.add_edge("RefinerAgent", "SummarizeResult")
        workflow.add_edge("AcceptanceAgent", END)
        graph = workflow.compile()
        # workflow.add_edge("NewsGenerationAgent", END)
        return graph

    async def run(self):
        # logger.info(f"🤖 Running scenario: {self.scenario_k}")
        try:
            return await self.workflow_with_data_generation.ainvoke(
                self.state, {"recursion_limit": 1e10}
            )
        except asyncio.TimeoutError:
            logger.warning(f"⏱️ Scenario timed out after 10 seconds: {self.scenario_k}")
            return ""


async def arun(hypo):
    agent_instance = CodeSwitchingAgent(hypo)
    print(f"🔍 Running scenario: {hypo}")
    await agent_instance.run()


async def main(config):
    first_lang = config["pre_execute"]["first_language"]
    second_lang = config["pre_execute"]["second_language"]
    cs_ratio = config["pre_execute"]["cs_ratio"]
    scenarios: list[AgentRunningState] = [
    AgentRunningState(hypothesis=hypo,first_language=first_lang, second_language=second_lang,cs_ratio=cs_ratio)
    for hypo in generate_hypo_list()]

    # make a for loop, each loop run 10 scenarios
    results_count = 0
    #for i in range(0, 1, 40): #og 0, 8000, 40
    tasks = [arun(scenario) for scenario in scenarios[start:end]]

    # 使用 asyncio.as_completed 來逐個等待任務完成
    try:
        for task in asyncio.as_completed(tasks, timeout=7200):
            result = await task
            print(result)
            results_count += 1
    except asyncio.TimeoutError:
        logger.warning(f"⏱️ Scenario timed out after 2400 seconds: {i}")
        print(f"🔍 Scenario timed out after 2400 seconds: {i}")
    finally:
        # log the number of results finished
        if results_count % 10 == 0:
            logger.info(f"🔍 Number of results finished: {results_count}")
            print(f"🔍 Number of results finished: {results_count}")
    return results_count

if __name__ == "__main__":
    try:
        asyncio.run(main(config))
    except Exception as e:
        logger.error(f"🚨 Error: {e}")
    # config: dict = load_config()
    # scenarios: list[AgentRunningState] = generate_scenarios(
    #     config["pre_execute"]
    # )
    # print(len(scenarios))

    # all_results 包含了所有 scenario 的结果

import yaml
from pprint import pprint
from node_models import AgentRunningState
import jsonlines as jsl
import os
import json
from read_xnli_dataset import XNLIDataLoader

def load_config(config_path: str):
    # Each config file will generate a different scenarios ~1440
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config

def create_hypo_json():
    # load xnli dataset
    loader = XNLIDataLoader(lang='en', test_path='xnli.test.tsv')
    # get list of hypo
    hypo_list = loader.get_hypotheses_json()
    # save hypo json
    loader.save_to_json("xnli_hypo.json")
    return hypo_list

def load_hypo_json():
    with open("xnli_hypo.json", "r", encoding="utf-8") as f:
        return json.load(f)
    
def generate_hypo_list():
    if os.path.isfile("xnli_hypo.json"):
        hypo_list=load_hypo_json()
    else:
        hypo_list=create_hypo_json()
    return hypo_list
    

def weighting_scheme(state):
    fluency = state["fluency_result"]["fluency_score"]
    naturalness = state["naturalness_result"]["naturalness_score"]
    csratio = state["cs_ratio_result"]["ratio_score"]
    socio = state["social_cultural_result"]["socio_cultural_score"]
    return fluency * 0.3 + naturalness * 0.25 + csratio * 0.2 + socio * 0.25


if __name__ == "__main__":
    if os.path.isfile("xnli_hypo.json"):
        hypo_list=create_hypo_json()
    else:
        hypo_list=load_hypo_json()
    # For a quick peek, let's print the first few
    for i, h in enumerate(hypo_list[:10]):
        print(f"hypo #{i+1}:", h)
        print("\n")

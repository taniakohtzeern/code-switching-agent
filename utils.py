import yaml
from pprint import pprint
from node_models import AgentRunningState
import jsonlines as jsl
import os
import csv
import json
import pandas as pd
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
    
def get_premise_label(loader: XNLIDataLoader):
    mapping = {}
    for idx, row in loader.data.iterrows():
        hypo = row['hypo']
        premise = row['premise']
        label = row['label']
        mapping[hypo] = {"premise": premise, "label": label}
    return mapping

def weighting_scheme(state):
    accuracy = state["accuracy_result"]["accuracy_score"]
    fluency = state["fluency_result"]["fluency_score"]
    naturalness = state["naturalness_result"]["naturalness_score"]
    # csratio = state["cs_ratio_result"]["ratio_score"]
    #socio = state["social_cultural_result"]["socio_cultural_score"]
    return accuracy*0.3 + fluency * 0.4 + naturalness * 0.3

def save_jsonl_to_tsv(jsonl_file, tsv_file, loader: XNLIDataLoader):
    """
    Read JSONL (code-switched outputs) and update/add rows into TSV.
    Matching uses (sentence1 == premise) AND (gold_label == label) to avoid
    overwriting other hypos that share the same premise.
    """

    # Load existing TSV if present
    if os.path.exists(tsv_file):
        try:
            df = pd.read_csv(tsv_file, sep="\t", dtype=str).fillna("")
        except Exception as e:
            print(f"[WARN] Could not read existing TSV '{tsv_file}': {e}")
            df = pd.DataFrame(columns=["sentence1", "sentence2", "gold_label"])
    else:
        df = pd.DataFrame(columns=["sentence1", "sentence2", "gold_label"])

    # Build mapping original_hypo -> {premise, label}
    mapping = {}
    for _, row in loader.data.iterrows():
        orig_hypo = row.get('hypo', "")
        premise = row.get('premise', "")
        label = row.get('label', "")
        if orig_hypo:
            mapping[orig_hypo] = {"premise": premise, "label": label}

    # Read JSONL; for each record update or append using (premise,label) key
    if os.path.exists(jsonl_file):
        with jsl.open(jsonl_file, 'r') as reader:
            for obj in reader:
                original_hypo = obj.get("hypothesis", {}).get("hypo", "")
                if not original_hypo:
                    continue

                # Robust extraction of translated sentence
                code_switched = obj.get("data_translation_result", "")
                if isinstance(code_switched, dict):
                    translated_sentence = (
                        code_switched.get("translated_sentence")
                        or code_switched.get("translation")
                        or ""
                    )
                else:
                    translated_sentence = str(code_switched)

                if not translated_sentence:
                    continue

                if original_hypo not in mapping:
                    print(f"[WARN] original hypo not in loader mapping: {original_hypo!r}")
                    continue

                premise = mapping[original_hypo]["premise"]
                label = mapping[original_hypo]["label"]

                # Find rows matching BOTH premise and gold_label
                matches = df.index[(df['sentence1'] == premise) & (df['gold_label'] == label)].tolist()

                if matches:
                    idx = matches[0]
                    # Update the matching row in-place
                    df.at[idx, 'sentence2'] = translated_sentence
                    df.at[idx, 'gold_label'] = label  # keep label consistent
                else:
                    # Append a new row (avoid repeated concat)
                    new_row = {"sentence1": premise, "sentence2": translated_sentence, "gold_label": label}
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Save updated TSV
    df.to_csv(tsv_file, sep="\t", index=False)
    print(f"Saved/updated {len(df)} rows to {tsv_file}")
    
if __name__ == "__main__":
    if os.path.isfile("xnli_hypo.json"):
        hypo_list=create_hypo_json()
    else:
        hypo_list=load_hypo_json()
    # For a quick peek, let's print the first few
    for i, h in enumerate(hypo_list[:10]):
        print(f"hypo #{i+1}:", h)
        print("\n")

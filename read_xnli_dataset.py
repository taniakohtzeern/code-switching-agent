import pandas as pd
import csv
import json

class XNLIDataLoader():
    """
    A PyTorch Dataset for XNLI (Cross-lingual Natural Language Inference) task.

    Supports train, dev, and test splits in a specific language, 
    tokenizes text inputs for GPT-style models, and optionally subsamples the dataset.

    Attributes:
        split (str): Dataset split, one of 'train', 'dev', 'test'.
        lang (str): Language code (e.g., 'en', 'zh').
        tokenizer: A HuggingFace tokenizer to convert text to input IDs.
        max_length (int): Maximum sequence length for tokenization.
        LABEL2ID (dict): Mapping from textual labels to integer IDs.
        ID2LABEL (dict): Reverse mapping from integer IDs to textual labels.
        data (pd.DataFrame): The loaded and preprocessed dataset.
    """
    def __init__(
        self,
        lang="en",
        test_path="xnli.test.tsv",
        max_length=1024,
        subset = 1.0  # 0~1
    ):
        self.lang = lang
        self.max_length = max_length
        self.LABEL2ID = {"entailment": 0, "contradictory": 1, "neutral": 2}
        self.ID2LABEL = {v: k for k, v in self.LABEL2ID.items()}

        path = test_path
        df = self.read_xnli_tsv(path)
        df = df[df['language']==lang].copy()
        keep_cols = ['sentence1', 'sentence2', 'gold_label']
        df = df[keep_cols].dropna()
        df.rename(columns={'sentence1':'premise','sentence2':'hypo','gold_label':'label'}, inplace=True)
        df['label'] = df['label'].replace({'contradiction': 'contradictory'})
    
        original_num = len(df)
        if subset < 1.0:
            n = max(1, int(len(df) * subset))
            df = df.iloc[:n].reset_index(drop=True)
        subset_num = len(df)

        self.data = df.reset_index(drop=True)
        print(f"Dataset initialized:  lang='{lang}', total={original_num}, subset={subset}, subset_count={subset_num}")

    def read_xnli_tsv(self, path):
        """
        Read an XNLI TSV file and return it as a pandas DataFrame.

        Args:
            path (str): Path to the TSV file.
            split (str): One of "train", "dev", "test" indicating the dataset split.

        Returns:
            pd.DataFrame: The dataset as a DataFrame with appropriate columns.
        """

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            rows = list(reader)
        header = rows[0]
        expected_cols = len(header)
        data = []
        for i, row in enumerate(rows[1:], start=2):
            if len(row) == expected_cols:
                data.append(row)
            else:
                print(f"skip row {i}: {len(row)} cols → {row[:2]}")
        return pd.DataFrame(data, columns=header)
    def get_hypotheses_json(self):
        """
        Returns the data as a list of dictionaries containing only the 'hypo' key.
        Format: [{"hypo": "sentence A..."}, {"hypo": "sentence B..."}]
        """
        # Select only the 'hypo' column and convert to records
        json_data = self.data[['hypo']].to_dict(orient='records')
        return json_data

    def save_to_json(self, output_file="output.json"):
        """Saves the hypotheses directly to a file."""
        data = self.get_hypotheses_json()
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(data)} hypotheses to {output_file}")

    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.data)
   

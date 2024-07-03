from langchain_community.llms import Ollama
import re
from enum import Enum
import pandas as pd
import polars as pl
from tqdm import tqdm
import logging
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
from dataclasses import dataclass

@dataclass
class Results:
    labels: list[str]
    cm: pl.DataFrame
    f1_scores: pl.DataFrame
    data: pl.DataFrame

class Experiment:
    class RelationType(Enum):
        def from_response(response: str):
            pass

        def prompt_option() -> str:
            pass

        def get_labels() -> list[str]:
            pass

    
    class BinaryRelationType(RelationType):
        SUPPORT = "support"
        ATTACK = "attack"

        def from_response(response: str):
            if re.search("support", response, re.IGNORECASE):
                return Experiment.BinaryRelationType.SUPPORT
            elif re.search("attack", response, re.IGNORECASE):
                return Experiment.BinaryRelationType.ATTACK
            else:
                raise Experiment.LLMRelationPredictionResponseError(response)
            
        def prompt_option() -> str:
            return "respond either \"support\" or \"attack\""
        
        def get_labels() -> list[str]:
            return ["support", "attack"]

    class TernaryRelationType(RelationType):
        SUPPORT = "support"
        ATTACK = "attack"
        UNRELATED = "unrelated"

        def from_response(response: str):
            if re.search("support", response, re.IGNORECASE):
                return Experiment.TernaryRelationType.SUPPORT
            elif re.search("attack", response, re.IGNORECASE):
                return Experiment.TernaryRelationType.ATTACK
            elif re.search("unrelated", response, re.IGNORECASE):
                return Experiment.TernaryRelationType.UNRELATED
            else:
                raise Experiment.LLMRelationPredictionResponseError(response)
            
        def prompt_option() -> str:
            return "respond either \"support\", \"attack\", or \"unrelated\""

        def get_labels() -> list[str]:
            return ["support", "attack", "unrelated"]

    class LLMResponseError(Exception):
        def __init__(self, message, response):
            super().__init__(message)

            self.message = message
            self.response = response

        def __str__(self):
            return f"{self.message}, got: \"{self.response}\""

    class LLMRelationPredictionResponseError(LLMResponseError):
        def __init__(self, response):
            super().__init__("Unexpected LLM response value", response)
    

    def __init__(self, model: str, output_dir: str = ".", relation_dim: str = "binary"):
        self.llm = Ollama(
            model=model
        )
        self.output_dir = output_dir

        if relation_dim == "binary":
            self.relation_dim = Experiment.BinaryRelationType
        else:
            self.relation_dim = Experiment.TernaryRelationType

    
    def __predict_relation(self, arg_1: str, arg_2: str) -> RelationType:
        prompt = f"""
            What's the relation between those two arguments? {self.relation_dim.prompt_option()}

            Arg1: {arg_1}
            Arg2: {arg_2}
            Relation: 
        """
        
        response = self.llm.invoke(prompt)

        return self.relation_dim.from_response(response)
        
    def run_from_df(self, df: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
        # Iterates through the rows to make LLM predict each argument pair relation
        predictions: list[Experiment.RelationType] = []
        for row in tqdm(df.to_numpy(), desc="predicted_relation"):
            # Make at least 5 attempts to predict relation as LLM
            # could fail to follow template
            for attempt in range(5):
                try:
                    prediction = self.__predict_relation(row[0], row[1])
                except Experiment.LLMResponseError as e:
                    logging.warning(f"attempt={attempt} {e}")
                    continue
                else:
                    predictions.append(prediction.value)
                    break
            else:
                raise Exception("LLM failed to predict relation after 5 attempts")

        # Add the prediction back to the DataFrame and convert it as a Pandas DataFrame
        results = df.with_columns(pl.Series("predicted_relation", predictions)).to_pandas()
    
        # Create and return the confusion matrix plus its labels
        labels = self.relation_dim.get_labels()
        cm = confusion_matrix(results['relation'], results['predicted_relation'], labels=labels)
        cm_df = pl.from_pandas(pd.DataFrame(cm, index=labels, columns=labels))
        f1_scores = f1_score(results['relation'], results['predicted_relation'], labels=labels, average=None)
        f1_scores_df = pl.from_pandas(pd.DataFrame(f1_scores, index=labels))

        return Results(labels, cm_df, f1_scores_df, results)

    def run_from_csv(self, input_file: str) -> tuple[np.ndarray, list[str]]:
        # Load .csv input file into DataFrame
        df = pl.read_csv(input_file)

        return self.run_from_df(df)
    
    def run_from_json(self, input_file: str) -> tuple[np.ndarray, list[str]]:
        # Load .json input file into DataFrame
        df = pl.read_json(input_file)

        return self.run_from_df(df)

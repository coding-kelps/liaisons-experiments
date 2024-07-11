from langchain_community.llms import Ollama
import re
from enum import Enum
import polars as pl
import pandas as pd
from tqdm import tqdm
import logging
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
from dataclasses import dataclass

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
    

    def __init__(self, model: str, output_dir: str = ".", tqdm = tqdm):
        self.llm = Ollama(
            model=model
        )
        self.output_dir = output_dir
        self.tqdm = tqdm
    
    def __predict_relation(self, arg_1: str, arg_2: str) -> RelationType:
        prompt = f"""
            What's the relation between those two arguments? {self.relation_dim.prompt_option()}

            Arg1: {arg_1}
            Arg2: {arg_2}
            Relation: 
        """
        
        response = self.llm.invoke(prompt)

        return self.relation_dim.from_response(response)
        
    def run_from_df(self, df: pl.DataFrame, relation_dim: str = "binary") -> pl.DataFrame:
        if relation_dim == "binary":
            self.relation_dim = Experiment.BinaryRelationType
        else:
            self.relation_dim = Experiment.TernaryRelationType

        # Iterates through the rows to make LLM predict each argument pair relation
        predictions: list[Experiment.RelationType] = []
        for row in self.tqdm(df[["argument_a", "argument_b"]].to_numpy(), desc=f"predict argument relation - model={self.llm.model} relation_dimension={relation_dim}"):
            # Make at least 5 attempts to predict relation as LLM
            # could fail to follow template
            for attempt in range(5):
                try:
                    prediction = self.__predict_relation(row[0], row[1])
                except Experiment.LLMResponseError as e:
                    logging.warning(f"attempt={attempt} model={self.llm.model} relation_dimension={relation_dim} msg={e}")
                    continue
                else:
                    predictions.append(prediction.value)
                    break
            else:
                logging.error(f"model={self.llm.model} relation_dimension={relation_dim} msg=LLM failed to follow prediction response template after 5 attempts")

        # Add the prediction back to the DataFrame and return it
        return df.with_columns(pl.Series("predicted_relation", predictions))
    
    def run_from_csv(self, input_file: str, relation_dim: str = "binary") -> pl.DataFrame:
        # Load .csv input file into DataFrame
        df = pl.read_csv(input_file)

        return self.run_from_df(df, relation_dim)
    
    def run_from_json(self, input_file: str, relation_dim: str = "binary") -> pl.DataFrame:
        # Load .json input file into DataFrame
        df = pl.read_json(input_file)

        return self.run_from_df(df, relation_dim)
    
    @dataclass
    class Benchmarks:
        confusion_matrix: pd.DataFrame
        f1_scores: pd.DataFrame

    def compute_benchmarks(self, results: pl.DataFrame) -> Benchmarks:
        # Retrieve labels from relation dimension
        labels = self.relation_dim.get_labels()

        # Compute confusion matrix
        cm = confusion_matrix(results['relation'], results['predicted_relation'], labels=labels)

        # Compute F1 scores for each prediction class
        f1_scores = f1_score(results['relation'], results['predicted_relation'], labels=labels, average=None)

        # Flip each F1 score as a column and convert it as a Pandas dataframe to enhance readability
        f1_scores_df = pd.DataFrame(np.split(f1_scores, True), index=[self.llm.model], columns=labels)

        # Add Macro F1 score as a separate column
        f1_scores_df = f1_scores_df.assign(macro=f1_scores_df.mean(axis=1))

        return Experiment.Benchmarks(
            # converted as a Pandas dataframe to enhance readability
            confusion_matrix=pd.DataFrame(cm, index=labels, columns=labels),
            f1_scores=f1_scores_df,
        )
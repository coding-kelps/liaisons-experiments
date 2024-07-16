import re
from enum import Enum
import pandas as pd
from tqdm import tqdm
import logging
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
from dataclasses import dataclass
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import GoogleGenerativeAI
from typing import Callable

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
    

    def __init__(self, llm, output_dir: str = ".", tqdm = tqdm):
        self.llm = llm
        self.output_dir = output_dir
        self.tqdm = tqdm

        if isinstance(llm, ChatOpenAI):
            self.client = "openai"
            self.model_name = llm.model_name
        elif isinstance(llm, ChatAnthropic):
            self.client = "anthropic"
            self.model_name = llm.model
        elif isinstance(llm, GoogleGenerativeAI):
            self.client = "google"
            self.model_name = llm.model
        elif isinstance(llm, ChatOllama):
            self.client = "ollama"
            self.model_name = llm.model
        else:
            self.client = "unknown"
            self.model_name = "unknown"
    
    def __predict_relation(self, arg_1: str, arg_2: str, prompt_formater: Callable[[str, str], str]) -> RelationType:
        prompt = prompt_formater(arg_1, arg_2)

        response = self.llm.invoke(prompt)

        if isinstance(self.llm, GoogleGenerativeAI):
            res_content = response
        else:
            res_content = response.content

        return self.relation_dim.from_response(res_content)
        
    def run_from_df(self, df: pd.DataFrame, prompt_formater: Callable[[str, str], str], relation_dim: str = "binary") -> pd.DataFrame:
        if relation_dim == "binary":
            self.relation_dim = Experiment.BinaryRelationType
        else:
            self.relation_dim = Experiment.TernaryRelationType

        # Iterates through the rows to make LLM predict each argument pair relation
        predictions: list[Experiment.RelationType] = []
        for i, row in enumerate(self.tqdm(df[["argument_a", "argument_b"]].to_numpy(), desc=f"Predict Argument Relation - client={self.client} model={self.model_name} relation_dimension={relation_dim}")):
            # Make at least 5 attempts to predict relation as LLM
            # could fail to follow template
            for attempt in range(5):
                try:
                    prediction = self.__predict_relation(row[0], row[1], prompt_formater)
                except Experiment.LLMResponseError as e:
                    logging.debug(f"client={self.client} model={self.model_name} relation_dimension={relation_dim} arg_pair_id={i} attempt={attempt} msg={e}")
                    continue
                else:
                    predictions.append(prediction.value)
                    break
            else:
                predictions.append("prediction_failed")
                logging.error(f"client={self.client} model={self.model_name} relation_dimension={relation_dim} arg_pair_id={i} msg=LLM failed to follow prediction response template after 5 attempts")

        # Add the prediction back to the DataFrame and return it
        df["predicted_relation"] = predictions

        return df
    
    def run_from_csv(self, input_file: str, prompt_formater: Callable[[str, str], str], relation_dim: str = "binary") -> pd.DataFrame:
        # Load .csv input file into DataFrame
        df = pd.read_csv(input_file)

        return self.run_from_df(df, prompt_formater, relation_dim)
    
    def run_from_json(self, input_file: str, prompt_formater: Callable[[str, str], str], relation_dim: str = "binary") -> pd.DataFrame:
        # Load .json input file into DataFrame
        df = pd.read_json(input_file)

        return self.run_from_df(df, prompt_formater, relation_dim)
    
    @dataclass
    class Benchmarks:
        confusion_matrix: pd.DataFrame
        f1_scores: pd.DataFrame
        metadata: pd.DataFrame

    def compute_benchmarks(self, results: pd.DataFrame) -> Benchmarks:
        # Retrieve labels from relation dimension
        labels = self.relation_dim.get_labels()

        # Remove failed prediction from results
        clean_results = results[results["predicted_relation"] != "prediction_failed"]

        metadata = pd.DataFrame({
            # Compute the prediction failing rate over the number of prediction
            "fail_rate": [results[results["predicted_relation"] == "prediction_failed"].shape[0] / results.shape[0]]
        })

        # Compute confusion matrix
        cm = confusion_matrix(clean_results['relation'], clean_results['predicted_relation'], labels=labels)

        # Compute F1 scores for each prediction class
        f1_scores = f1_score(clean_results['relation'], clean_results['predicted_relation'], labels=labels, average=None)

        # Flip each F1 score as a column and convert it as a Pandas dataframe to enhance readability
        f1_scores_df = pd.DataFrame(np.split(f1_scores, True), index=[self.model_name], columns=labels)

        # Add Macro F1 score as a separate column
        f1_scores_df = f1_scores_df.assign(macro=f1_scores_df.mean(axis=1))

        return Experiment.Benchmarks(
            # converted as a Pandas dataframe to enhance readability
            confusion_matrix=pd.DataFrame(cm, index=labels, columns=labels),
            f1_scores=f1_scores_df,
            metadata=metadata,
        )

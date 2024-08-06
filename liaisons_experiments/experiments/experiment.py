import re
from enum import Enum
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
from dataclasses import dataclass
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import GoogleGenerativeAI
from typing import Callable
import concurrent.futures

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
    

    def __init__(self, llm, output_dir: str = ".", num_workers: int = 1, tqdm = tqdm):
        self.llm = llm
        self.output_dir = output_dir
        self.tqdm = tqdm
        self.num_workers = num_workers

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
    
    def __predict_relation(self, parent_argument: str, child_argument: str, prompt_formater: Callable[[str, str], str], pbar: tqdm) -> str | None:
        prompt = prompt_formater(parent_argument, child_argument)

        for _ in range(5):
            try:
                response = self.llm.invoke(prompt)

                if isinstance(self.llm, GoogleGenerativeAI):
                    res_content = response
                else:
                    res_content = response.content
            
                prediction = self.relation_dim.from_response(res_content)

            except Experiment.LLMResponseError as _:
                continue
            else:
                pbar.update(1)
                return prediction.value
        else:
            pbar.update(1)
            return "prediction_failed"
        
    def run_from_df(self, df: pd.DataFrame, prompt_formater: Callable[[str, str], str], relation_dim: str = "binary") -> pd.DataFrame:        
        if relation_dim == "binary":
            self.relation_dim = Experiment.BinaryRelationType
        else:
            self.relation_dim = Experiment.TernaryRelationType


        arg_pairs = df[["parent_argument", "child_argument"]].to_numpy()

        with self.tqdm(total=arg_pairs.shape[0], desc=f"Predict Argument Pair Relation", unit="pred", postfix={"model": self.model_name, "relation_dimension": relation_dim, "num_workers": self.num_workers}) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                df["predicted_relation"] = list(executor.map(lambda row: self.__predict_relation(row[0], row[1], prompt_formater, pbar), arg_pairs))
            
            pbar.close()

        return df
    
    def run_from_csv(self, input_file: str, **kwargs) -> pd.DataFrame:
        # Load .csv input file into DataFrame
        df = pd.read_csv(input_file)

        return self.run_from_df(df, **kwargs)
    
    def run_from_json(self, input_file: str, **kwargs) -> pd.DataFrame:
        # Load .json input file into DataFrame
        df = pd.read_json(input_file)

        return self.run_from_df(df, **kwargs)
    
    @dataclass
    class Benchmarks:
        confusion_matrix: pd.DataFrame
        f1_scores: pd.DataFrame
        metadata: pd.DataFrame

    def compute_benchmarks(self, results: pd.DataFrame) -> Benchmarks:
        # Retrieve labels from relation dimension
        labels = self.relation_dim.get_labels()

        metadata = pd.DataFrame({
            # Compute the prediction failing rate over the number of prediction
            "fail_rate": [results[results["predicted_relation"] == "prediction_failed"].shape[0] / results.shape[0]],
            "model_name": [self.model_name],
        })

        # Compute confusion matrix
        cm = confusion_matrix(results['relation'], results['predicted_relation'], labels=labels)

        # Compute F1 scores for each prediction class
        f1_scores = f1_score(results['relation'], results['predicted_relation'], labels=labels, average=None)

        # Flip each F1 score as a column and convert it as a Pandas dataframe to enhance readability
        f1_scores_df = pd.DataFrame(np.split(f1_scores, True), columns=labels)

        # Add Macro F1 score as a separate column
        f1_scores_df = f1_scores_df.assign(macro=f1_scores_df.mean(axis=1), model_name=self.model_name).reset_index(drop=True)

        return Experiment.Benchmarks(
            # converted as a Pandas dataframe to enhance readability
            confusion_matrix=pd.DataFrame(cm, index=labels, columns=labels),
            f1_scores=f1_scores_df,
            metadata=metadata,
        )

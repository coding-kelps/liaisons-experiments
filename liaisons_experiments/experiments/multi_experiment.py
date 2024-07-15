from .experiment import Experiment
from dataclasses import dataclass
from tqdm import tqdm
import pandas as pd
from typing import Callable
from datetime import datetime

class MultiExperiment:
    def __init__(self, llms: list, output_dir: str = ".", tqdm = tqdm):
        self.experiments: list[Experiment] = []

        for llm in llms:
            self.experiments.append(Experiment(llm, output_dir, tqdm))
    
    @dataclass
    class Benchmarks:
        confusion_matrices: dict[str, pd.DataFrame]
        f1_scores: pd.DataFrame
        metadata: pd.DataFrame
    
    def run_from_csv(self, input_file: str, prompt_formater: Callable[[str, str, list[str]], str], relation_dim: str = "binary") -> Benchmarks:
        str_date = datetime.today().strftime("%Y %m %d")

        # Load .csv input file into DataFrame
        df = pd.read_csv(input_file)

        benchmarks = MultiExperiment.Benchmarks({}, pd.DataFrame(), pd.DataFrame())

        for experiment in self.experiments:
            results = experiment.run_from_df(df, prompt_formater, relation_dim)
            exp_benchmarks = experiment.compute_benchmarks(results)

            benchmarks.confusion_matrices[experiment.model_name] = exp_benchmarks.confusion_matrix
            benchmarks.f1_scores = pd.concat([benchmarks.f1_scores, exp_benchmarks.f1_scores], axis=0)
            benchmarks.metadata = pd.concat([benchmarks.metadata, exp_benchmarks.metadata], axis=0)

            # Gradually save benchmarks to file
            benchmarks.f1_scores.to_csv(f"{self.output_dir}/results/{str_date}/{relation_dim}_benchmarks_f1_score.csv")
            benchmarks.metadata.to_csv(f"{self.output_dir}/results/{str_date}/{relation_dim}_benchmarks_metadata.csv")

        return benchmarks

from .experiment import Experiment
from dataclasses import dataclass
import pandas as pd
from typing import Callable
from datetime import datetime
from names_generator import generate_name
import logging
import os

class MultiExperiment:
    def __init__(self, exps: list, output_dir: str = ".", **kwargs):        
        self.experiments: list[Experiment] = list(map(lambda exp: Experiment(exp[0], **{**kwargs, **exp[1]}), exps))
        self.output_dir = output_dir

    @dataclass
    class Benchmarks:
        confusion_matrices: dict[str, pd.DataFrame]
        f1_scores: pd.DataFrame
        metadata: pd.DataFrame

    def run_from_df(self, df: pd.DataFrame, prompt_formater: Callable[[str, str, list[str]], str], relation_dim: str = "binary", name: str | None = None, input_name: bool = False,) -> Benchmarks:
        if not name and input_name:
            name = input()
        elif not name:
            name = generate_name()

        # Create output directories if none
        str_date = datetime.today().strftime("%Y-%m-%d")
        full_output_dir = f"{self.output_dir}/results/{str_date}/{name}"

        if not os.path.exists(full_output_dir):
            os.makedirs(full_output_dir)
            print(f"created multi-experiments output directory \"{name}\"")

        benchmarks = MultiExperiment.Benchmarks({}, pd.DataFrame(), pd.DataFrame())

        for experiment in self.experiments:
            results = experiment.run_from_df(df, prompt_formater, relation_dim)
            exp_benchmarks = experiment.compute_benchmarks(results)

            benchmarks.confusion_matrices[experiment.model_name] = exp_benchmarks.confusion_matrix
            benchmarks.f1_scores = pd.concat([benchmarks.f1_scores, exp_benchmarks.f1_scores], axis=0)
            benchmarks.metadata = pd.concat([benchmarks.metadata, exp_benchmarks.metadata], axis=0)

            # Gradually save benchmarks to file
            benchmarks.f1_scores.to_csv(f"{full_output_dir}/{relation_dim}_benchmarks_f1_score.csv", index=False)
            benchmarks.metadata.to_csv(f"{full_output_dir}/{relation_dim}_benchmarks_metadata.csv", index=False)

        return benchmarks

    def run_from_csv(self, input_file: str, **kwargs) -> Benchmarks:
        # Load .csv input file into DataFrame
        df = pd.read_csv(input_file)

        return self.run_from_df(df, **kwargs)

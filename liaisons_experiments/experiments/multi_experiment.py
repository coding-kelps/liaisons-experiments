from .experiment import Experiment
from dataclasses import dataclass
from tqdm import tqdm
import pandas as pd
import polars as pl

@dataclass
class ExperimentConfig:
    model: str

class MultiExperiment:
    def __init__(self, configs: list[ExperimentConfig], output_dir: str = ".", tqdm = tqdm):
        self.experiments: list[Experiment] = []

        for config in configs:
            self.experiments.append(Experiment(config.model, output_dir, tqdm))
    
    @dataclass
    class Benchmarks:
        confusion_matrices: dict[str, pd.DataFrame]
        f1_scores: pd.DataFrame
    
    def run_from_csv(self, input_file: str, relation_dim: str = "binary") -> Benchmarks:
        # Load .csv input file into DataFrame
        df = pl.read_csv(input_file)

        benchmarks = MultiExperiment.Benchmarks({}, pd.DataFrame())

        for experiment in self.experiments:
            results = experiment.run_from_df(df, relation_dim)
            exp_benchmarks = experiment.compute_benchmarks(results)

            benchmarks.confusion_matrices[experiment.llm.model] = exp_benchmarks.confusion_matrix
            benchmarks.f1_scores = pd.concat([benchmarks.f1_scores, exp_benchmarks.f1_scores], axis=0)

        return benchmarks

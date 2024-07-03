import unittest
from liaisons_experiments.experiments import Experiment
import polars as pl
import os

class TestExperiment(unittest.TestCase):
    def test_experiment_doesnt_crash(self):
        model = os.environ.get("LIAISONS_EXPERIMENTS_MODEL", "gemma:2b")
        output_dir = os.environ.get("LIAISONS_EXPERIMENTS_OUTPUT_DIR", ".")

        exp = Experiment(model, output_dir)

        # All the following arguments pairs examples have been taken from the research paper:
        # "A Dataset Independent Set of Baselines for Relation Prediction in Argument Mining"
        # Oana Cocarascu, Elena Cabrio, Serena Villata, Francesca Toni
        # doi: https://doi.org/10.48550/arXiv.2003.04970
        df = pl.from_dict({
            "argument_a": [
                "Research studies have yielded the conclusion that the effect of violent media consumption on aggressive be- havior is in the same ballpark statistically as the effect of smoking on lung cancer, the effect of lead exposure on children’s intellectual development and the effect of asbestos on laryngeal cancer",
                "People know video game violence is fake",
                "Children with many siblings receive fewer resources",
                "Virtually all developed countries today successfully promoted their national industries through protectionism",
                "i agree did not like this either in fact i stopped watching once waltz was killed because i just didnt care anymore",
                "samsung note it has a bigger screen and a somewhat faster processor",
                "We should implement Zoho, because it is cheaper than MS Office"
            ],
            "argument_b": [
                "Violent video games are real danger to young minds",
                "Youth playing violent games exhibit more aggression",
                "This house supports the one-child policy of the republic of China",
                "This house would unleash the free market",
                "after all the attention and awards etc and an imdb rating of i was so shocked to finally see this film and have it be so bad",
                "htc one it is currently the best one in the market good quality superb specs",
                "We should implement OpenOffice"
            ],
            "relation": [
                "support",
                "attack",
                "support",
                "attack",
                "support",
                "attack",
                "attack",
            ]
        })

        results = exp.run_from_df(df)
        

if __name__ == '__main__':
    unittest.main()

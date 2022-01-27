#!/usr/bin/env python
"""Provides Generic Classes to make an image analysis.

"""

from abc import ABC, abstractmethod

import pandas as pd


class InputData(ABC):
    def __init__(self, data):
        self._content = data

    @abstractmethod
    def read(self):
        pass


class Cohort(InputData):
    def __init__(self, dataframe, workdir=None):
        super().__init__(dataframe)
        self.workdir = workdir

    def read(self):
        for _, row in self._content.iterrows():
            filepath = row.path
            name = row.id
            if row.todo == 1 and filepath != 0:
                if self.workdir:
                    filepath = str(self.workdir / filepath)
                    print(type(filepath))
                yield (name, filepath)


class AnalysisCV(object):
    """ """

    def __init__(self, procedure):
        self.procedure = procedure

    def run(self, input_data):
        print("running analysis !!")

        all_results = {}

        for (name, filepath) in input_data.read():
            result = self.procedure.run(filepath, name)
            results_df = pd.DataFrame(result, columns=result[0].keys())
            all_results[name] = results_df
            results_df.to_csv(name + ".csv")
        return all_results

#!/usr/bin/env python
"""Provides Generic Classes to describe image analysis procedure.

"""

from abc import ABC, abstractmethod

__author__ = "Kevin Cortacero"
__copyright__ = "Copyright 2020, Cancer Image Analysis"
__credits__ = ["Kevin Cortacero"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Kevin Cortacero"
__email__ = "kevin.cortacero@inserm.fr"
__status__ = "Production"


class Procedure(ABC):
    def __init__(self, procedure_name):
        self._name = procedure_name

    @abstractmethod
    def run(self, filepath):
        pass

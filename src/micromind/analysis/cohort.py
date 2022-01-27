#!/usr/bin/env python
"""Provides Generic Classes to represent cohort data.

"""

__author__ = "Kevin Cortacero"
__copyright__ = "Copyright 2020, Cancer Image Analysis"
__credits__ = ["Kevin Cortacero"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Kevin Cortacero"
__email__ = "kevin.cortacero@inserm.fr"
__status__ = "Production"


class Patient(object):
    def __init__(self, data_path):
        self.data_path = data_path


class ChromogenicCohort(object):
    def __init__(self, metadata):
        self.metadata = metadata
        self.patients = {}
        self.setup()

    def setup(self):
        for index, row in self.metadata.iterrows():
            slide_path = row.PATH
            slide_name = row[0]
            if row.TODO == 1 and slide_path != 0:
                self.patients[slide_name] = Patient(slide_path)
        print(self.patients)

    def get_slides(self):
        for patient_id, patient in self.patients.items():
            yield ChromogenicSlide(patient.data_path, patient_id)

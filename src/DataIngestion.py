import os
import pandas as pd
from dataclasses import dataclass


@dataclass
class DataIngestionPathConfig:
    data_path = "NoteBooks/DATA/predictive_maintenance.csv"


class DataIngestion:
    def __init__(self):
        self.path = DataIngestionPathConfig.data_path

    def load_data(self) -> pd.DataFrame:
        """
        Loads the data using pandas.

        :return: Returns the raw data.
        """
        try:
            raw_data = pd.read_csv(self.path)
            return raw_data
        except Exception as e:
            print(f"Error occurred while loading data: {e}")
            return None

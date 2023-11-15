import pandas as pd

class CSVHandler:
    def __init__(self, input_csv):
        self.input_csv = input_csv
        self.df = self._read_csv()

    def _read_csv(self):
        df = pd.read_csv(self.input_csv)
        df = df.dropna(how='all')
        return df

    def get_dataframe(self):
        return self.df


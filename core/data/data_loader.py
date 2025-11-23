import pandas as pd


class DataLoader:
    def __init__(self, data_source):
        self.data_source = data_source

    def load_data(self, file_path: str) -> pd.DataFrame:
        data = pd.read_csv(file_path)
        return data

    def save_data(self, data: pd.DataFrame, file_path: str) -> None:
        data.to_csv(file_path, index=False)

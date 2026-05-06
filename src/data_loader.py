import pandas as pd

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise Exception("File not found!")

    if df.empty:
        raise Exception("Empty dataset!")

    return df
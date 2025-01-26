from difflib import SequenceMatcher
import json
import os


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def load_data(file_path: str = "../data/booksy_wroclaw_geocoded.json") -> dict:
    data_path = os.path.join(os.path.dirname(__file__), file_path)
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

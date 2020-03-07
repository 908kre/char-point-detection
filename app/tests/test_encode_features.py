import pytest
import pandas as pd
from app.encoders import parse_age, parse_floor
test_df = pd.read_csv("/store/data/test_data.csv")
print(test_df["間取り"].unique())


@pytest.mark.parametrize("test_input,expected", [
    ("昭和30年", 1955),
    ("平成3年", 1991),
    ("平成10年", 1998),
    (None, None)
])
def test_parse_age(test_input, expected):
    assert expected == parse_age(test_input)

@pytest.mark.parametrize("test_input,expected", [
    ("１K", {"dinning":0, "kitchen": 1, "room": 1, "living": 0, "storage": 0}),
    ("２LDK", {"dinning":1, "kitchen": 1, "room": 2, "living": 1, "storage": 0}),
    ("１K＋S", {"dinning":0, "kitchen": 1, "room": 1, "living": 0, "storage": 1}),
    ("１R", {"dinning":0, "kitchen": 0, "room": 1, "living": 0, "storage": 0}),
    ("スタジオ", {"dinning":0, "kitchen": 0, "room": 1, "living": 0, "storage": 0}),
])
def test_parse_floor(test_input, expected):
    assert expected == parse_floor(test_input)

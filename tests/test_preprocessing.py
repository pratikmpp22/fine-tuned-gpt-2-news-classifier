import pandas as pd
import pytest

from src.data.preprocessing import validate_ag_news_frame


def test_validate_ag_news_frame_accepts_valid_labels():
    df = pd.DataFrame({"text": ["headline"], "label": [2]})
    validate_ag_news_frame(df)


def test_validate_ag_news_frame_rejects_missing_columns():
    df = pd.DataFrame({"text": ["headline"]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_ag_news_frame(df)


def test_validate_ag_news_frame_rejects_invalid_labels():
    df = pd.DataFrame({"text": ["headline"], "label": [9]})
    with pytest.raises(ValueError, match="invalid labels"):
        validate_ag_news_frame(df)

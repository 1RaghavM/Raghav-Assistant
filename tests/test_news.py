import pytest
from services.news import get_news

def test_get_news_returns_string():
    result = get_news("technology")
    assert isinstance(result, str)
    assert len(result) > 0

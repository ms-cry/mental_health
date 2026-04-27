from src.preprocessing.preprocessor import clean_text

def test_clean_text():
    text = "Hello!!! Visit http://test.com"
    cleaned = clean_text(text)
    assert "http" not in cleaned
    assert cleaned.islower()

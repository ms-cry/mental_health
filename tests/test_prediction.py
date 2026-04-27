from src.models.predict import predict

def test_prediction():
    texts = ["I feel very sad", "Life is great"]
    preds = predict(texts)
    assert len(preds) == 2

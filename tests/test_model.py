import joblib
import pandas as pd


def test_prediction():
    model = joblib.load("model.pkl")
    colonnes = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup", "Latitude", "Longitude",
    ]
    sample = pd.DataFrame(
        [[8.3, 41.0, 6.98, 1.02, 322.0, 2.55, 37.88, -122.23]],
        columns=colonnes,
    )
    prediction = model.predict(sample)
    assert prediction[0] > 0  # Vérifie que le prix est positif

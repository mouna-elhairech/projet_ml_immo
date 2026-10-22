import joblib
import mlflow
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Prediction_Immo_Californie")

data = fetch_california_housing(as_frame=True)
X = data.frame.drop("MedHouseVal", axis=1)
y = data.frame["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():
    n_estimators = 100
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("MAE", mae)
    joblib.dump(model, "model.pkl")
    print(f"MAE obtenue : {mae:.4f}")

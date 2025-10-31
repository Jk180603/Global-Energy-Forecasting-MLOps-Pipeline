import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

mlflow.set_experiment("energyglobal-forecast")

with mlflow.start_run():
    df = pd.read_parquet("/content/energyglobal/data/processed/energy_data.parquet")

    # ONLY COLUMNS THAT EXIST
    feature_cols = ['population', 'gdp', 'gdp_per_capita', 'growth_rate']
    target = 'primary_energy_consumption'

    df = df.dropna(subset=feature_cols + [target])

    X = df[feature_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)

    mlflow.log_metric("mae", mae)
    mlflow.xgboost.log_model(model, "model")
    model.save_model("/content/energyglobal/models/consumption_model.json")

    print(f"Training DONE – MAE = {mae:.2f}")

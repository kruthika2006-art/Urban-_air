from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def train_model(df):

    X = df[
        ["number_of_monitoring_stations"]
    ]

    y = df["aqi_value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    score = r2_score(
        y_test,
        preds
    )

    return score, model

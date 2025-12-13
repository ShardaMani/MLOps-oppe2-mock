import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

TARGET_COL = "target"

DATASETS = {
    0: "data/data_v0.csv",
    2: "data/poisoned_2_percent.csv",
    8: "data/poisoned_8_percent.csv",
    20: "data/poisoned_20_percent.csv",
}

mlflow.set_experiment("mlsecurity_data_poisoning")

for poisoning_level, path in DATASETS.items():
    df = pd.read_csv(path)

    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # identify columns
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object", "bool", "category"]).columns

    # preprocessing
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    with mlflow.start_run(run_name=f"poisoning_{poisoning_level}%"):
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        f1 = f1_score(y_test, preds, pos_label="yes")

        mlflow.log_param("poisoning_level", poisoning_level)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(pipeline, "model")

        print(f"Poisoning {poisoning_level}% → F1-score: {f1:.4f}")

import pandas as pd

DATA_PATH = "data/cleaned_supply_chain_data.csv"


def prepare_features():
    # Load cleaned dataset
    df = pd.read_csv(DATA_PATH)

    # Target column
    y = df["delay"]

    # Remove ID and delay_days to avoid data leakage
    X = df.drop(
        columns=["shipment_id", "delay", "delay_days"]
    )

    # Identify categorical and numerical columns
    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    print("Feature engineering completed!")
    print("Dataset shape:", X.shape)
    print("Numerical features:", numerical_columns)
    print("Categorical features:", categorical_columns)
    print("Target shape:", y.shape)

    return X, y, categorical_columns, numerical_columns


if __name__ == "__main__":
    prepare_features()
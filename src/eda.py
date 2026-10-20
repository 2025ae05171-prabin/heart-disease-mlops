
"""Download and inspect the UCI Heart Disease dataset."""

from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    dataset = fetch_ucirepo(id=45)
    X = dataset.data.features.copy()
    y = dataset.data.targets.iloc[:, 0].copy()

    X.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in X.columns
    ]

    y = pd.to_numeric(y, errors="coerce")
    valid = y.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    df = X.copy()
    df["target"] = (y > 0).astype(int)

    print("Dataset shape:", df.shape)
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nClass distribution:")
    print(df["target"].value_counts().sort_index())

    df.to_csv("heart_disease.csv", index=False)

    df.select_dtypes(include="number").hist(
        figsize=(14, 10), bins=20
    )
    plt.tight_layout()
    plt.savefig("feature_histograms.png")
    plt.close()

    plt.figure(figsize=(12, 9))
    sns.heatmap(
        df.select_dtypes(include="number").corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    plt.close()

    sns.countplot(data=df, x="target")
    plt.title("Heart Disease Class Distribution")
    plt.tight_layout()
    plt.savefig("class_distribution.png")
    plt.close()

    print("\nSaved CSV and three EDA plots.")


if __name__ == "__main__":
    main()

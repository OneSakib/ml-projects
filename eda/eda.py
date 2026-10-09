# STEP 0: IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("default")
sns.set_theme(style="whitegrid")

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

df = pd.read_csv("WineQT.csv")
print(df.head())


# ============================================================
# STEP 2: CHECK SHAPE
# ============================================================

df.shape

# ============================================================
# STEP 3: VIEW DATA
# ============================================================

print("\nFirst 5 Rows:")
print(df.head())

print("\nLast 5 Rows:")
print(df.tail())

print("\nRandom 5 Rows:")
print(df.sample(min(5, len(df)), random_state=42))


# ============================================================
# STEP 4: COLUMN NAMES AND DATA TYPES
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: COLUMNS AND DATA TYPES")
print("=" * 60)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()


# ============================================================
# STEP 5: STATISTICAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: DESCRIBE")
print("=" * 60)

print("\nNumerical Summary:")
print(df.describe().T)

print("\nAll Columns Summary:")
print(df.describe(include="all").T)

numeric_cols = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_cols = df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical Columns:", numeric_cols)
print("Categorical Columns:", categorical_cols)


# ============================================================
# STEP 6: CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: MISSING VALUES")
print("=" * 60)

missing_count = df.isnull().sum()

missing_percentage = (
    df.isnull().mean() * 100
).round(2)

missing_summary = pd.DataFrame({
    "Missing Count": missing_count,
    "Missing Percentage": missing_percentage
})

print(missing_summary)


# ============================================================
# STEP 7: CHECK DUPLICATE ROWS
# ============================================================

print("\n" + "=" * 60)
print("STEP 7: DUPLICATES")
print("=" * 60)

duplicate_count = df.duplicated().sum()

print("Duplicate Rows:", duplicate_count)

# Display duplicates for inspection
if duplicate_count > 0:
    print("\nDuplicate Records:")
    print(df[df.duplicated(keep=False)].head(10))


# ============================================================
# STEP 8: CHECK UNIQUE VALUES
# ============================================================

print("\n" + "=" * 60)
print("STEP 8: UNIQUE VALUES")
print("=" * 60)

for column in df.columns:
    print(f"\n{column}")
    print("Unique Count:", df[column].nunique())
    print("Sample Values:", df[column].unique()[:10])


# ============================================================
# STEP 9: CHECK CONSTANT COLUMNS
# ============================================================

print("\n" + "=" * 60)
print("STEP 9: CONSTANT COLUMNS")
print("=" * 60)

constant_cols = [
    col for col in df.columns
    if df[col].nunique(dropna=False) <= 1
]

print("Constant Columns:", constant_cols)


# ============================================================
# STEP 10: UNIVARIATE ANALYSIS - HISTOGRAMS
# ============================================================

print("\n" + "=" * 60)
print("STEP 10: HISTOGRAMS")
print("=" * 60)

if numeric_cols:
    df[numeric_cols].hist(
        figsize=(16, 12),
        bins=20,
        edgecolor="black"
    )

    plt.suptitle("Numerical Feature Distributions")
    plt.tight_layout()
    plt.show()


# ============================================================
# STEP 11: SKEWNESS ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("STEP 11: SKEWNESS")
print("=" * 60)

if numeric_cols:
    print(df[numeric_cols].skew().sort_values())


# ============================================================
# STEP 12: OUTLIER CHECK USING IQR
# ============================================================

print("\n" + "=" * 60)
print("STEP 12: POTENTIAL OUTLIERS")
print("=" * 60)

outlier_results = []

for col in numeric_cols:
    series = df[col].dropna()

    if series.empty:
        continue

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (
        (series < lower_bound) |
        (series > upper_bound)
    )

    outlier_results.append({
        "Column": col,
        "Lower Bound": lower_bound,
        "Upper Bound": upper_bound,
        "Outlier Count": int(outlier_mask.sum())
    })

outlier_df = pd.DataFrame(outlier_results)

print(outlier_df)


# ============================================================
# STEP 13: BOX PLOTS
# ============================================================

print("\n" + "=" * 60)
print("STEP 13: BOX PLOTS")
print("=" * 60)

for col in numeric_cols:
    plt.figure(figsize=(8, 4))

    sns.boxplot(x=df[col])

    plt.title(f"Box Plot - {col}")
    plt.tight_layout()
    plt.show()


# ============================================================
# STEP 14: CATEGORICAL FEATURE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("STEP 14: CATEGORICAL ANALYSIS")
print("=" * 60)

for col in categorical_cols:
    print(f"\nValue Counts for {col}:")
    print(df[col].value_counts(dropna=False).head(20))

    plt.figure(figsize=(9, 4))

    df[col].value_counts().head(20).plot(
        kind="bar"
    )

    plt.title(f"Category Counts - {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


# ============================================================
# STEP 15: BIVARIATE ANALYSIS - SCATTER PLOTS
# ============================================================

print("\n" + "=" * 60)
print("STEP 15: BIVARIATE ANALYSIS")
print("=" * 60)

# Change this if your dataset has a different target
target_col = "quality"

if target_col in df.columns:
    for col in numeric_cols:
        if col == target_col:
            continue

        plt.figure(figsize=(7, 4))

        sns.scatterplot(
            data=df,
            x=col,
            y=target_col,
            alpha=0.6
        )

        plt.title(f"{col} vs {target_col}")
        plt.tight_layout()
        plt.show()


# ============================================================
# STEP 16: PAIR PLOT
# ============================================================

print("\n" + "=" * 60)
print("STEP 16: PAIR PLOT")
print("=" * 60)

# Use a small subset to keep pairplot readable and efficient
pair_cols = numeric_cols[:5]

if len(pair_cols) >= 2:
    sns.pairplot(
        df[pair_cols].dropna().sample(
            min(500, len(df.dropna(subset=pair_cols))),
            random_state=42
        ),
        diag_kind="hist"
    )

    plt.show()


# ============================================================
# STEP 17: VIOLIN PLOT AND GROUP COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("STEP 17: GROUP COMPARISON")
print("=" * 60)

if target_col in df.columns:
    for col in numeric_cols:
        if col == target_col:
            continue

        plt.figure(figsize=(9, 5))

        sns.violinplot(
            data=df,
            x=target_col,
            y=col,
            inner="quartile",
            cut=0
        )

        plt.title(f"{col} Distribution by {target_col}")
        plt.tight_layout()
        plt.show()


# ============================================================
# STEP 18: CORRELATION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("STEP 18: CORRELATION MATRIX")
print("=" * 60)

if len(numeric_cols) >= 2:
    correlation_matrix = df[numeric_cols].corr()

    print(correlation_matrix.round(2))

    plt.figure(figsize=(14, 10))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5
    )

    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


# ============================================================
# STEP 19: TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("STEP 19: TARGET ANALYSIS")
print("=" * 60)

if target_col in df.columns:
    print(df[target_col].value_counts(dropna=False))

    plt.figure(figsize=(8, 4))

    sns.countplot(
        data=df,
        x=target_col
    )

    plt.title(f"Target Distribution - {target_col}")
    plt.tight_layout()
    plt.show()


# ============================================================
# STEP 20: FINAL EDA SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("STEP 20: FINAL SUMMARY")
print("=" * 60)

print("Dataset Shape:", df.shape)
print("Total Missing Values:", int(df.isnull().sum().sum()))
print("Total Duplicate Rows:", int(df.duplicated().sum()))
print("Total Numerical Columns:", len(numeric_cols))
print("Total Categorical Columns:", len(categorical_cols))
print("Constant Columns:", constant_cols)

print("\nEDA COMPLETED SUCCESSFULLY!")

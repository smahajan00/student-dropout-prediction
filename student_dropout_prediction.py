"""
COMP 30044 CW2
Predicting Student Dropout and Academic Success using Machine Learning.

This script performs the complete assignment workflow:
- load and inspect the UCI student dataset
- filter the task to Dropout vs Graduate
- create and save all required EDA plots
- train and evaluate the required machine learning models
- save report values and Streamlit app artifacts
"""

# =============================================================================
# Imports and configuration
# =============================================================================

from pathlib import Path
import os
import warnings

PROJECT_DIR = Path(__file__).resolve().parent
MPL_CACHE_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "student_dropout_prediction_matplotlib_cache"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message="FigureCanvasAgg is non-interactive, and thus cannot be shown",
    category=UserWarning,
)

FIGURES_DIR = PROJECT_DIR / "figures"
LOCAL_DATA_PATH = PROJECT_DIR / "data.csv"

UCI_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00697/"
    "predict_students_dropout_and_academic_success.csv"
)

RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET_COLUMN = "Target"
BINARY_TARGET_CLASSES = ["Dropout", "Graduate"]

ACADEMIC_COLUMNS_FOR_CORRELATION = [
    "Admission grade",
    "Previous qualification (grade)",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
]

SCALING_PREVIEW_COLUMNS = [
    "Admission grade",
    "Age at enrollment",
    "Previous qualification (grade)",
    "Curricular units 1st sem (approved)",
    "Curricular units 2nd sem (approved)",
]

MEDIAN_COLUMNS = [
    "Age at enrollment",
    "Admission grade",
    "Curricular units 1st sem (approved)",
    "Curricular units 2nd sem (approved)",
]

REQUIRED_COLUMNS = sorted(
    set(
        [TARGET_COLUMN]
        + ACADEMIC_COLUMNS_FOR_CORRELATION
        + SCALING_PREVIEW_COLUMNS
        + MEDIAN_COLUMNS
        + ["Tuition fees up to date", "Scholarship holder", "Debtor"]
    )
)

PLOT_FILENAMES = [
    "target_distribution_before_filtering.png",
    "target_distribution_after_filtering.png",
    "correlation_heatmap.png",
    "regression_admission_grade_first_sem_grade.png",
    "age_target_boxplot.png",
    "admission_grade_target_boxplot.png",
    "tuition_fees_target_countplot.png",
    "scholarship_target_countplot.png",
    "first_sem_approved_units_boxplot.png",
    "second_sem_approved_units_boxplot.png",
    "model_comparison_bar_chart.png",
    "confusion_matrix_tuned_random_forest.png",
    "feature_importance_tuned_random_forest.png",
]


# =============================================================================
# Helper functions
# =============================================================================


def print_section(title: str) -> None:
    """Print a readable terminal section heading."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def save_plot(filename: str) -> None:
    """Save the current plot into figures/ and display it for notebook-style runs."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    plt.show()
    plt.close()


def ordered_targets(df: pd.DataFrame) -> list[str]:
    """Return the binary target classes that are present in the dataframe."""
    return [target for target in BINARY_TARGET_CLASSES if target in df[TARGET_COLUMN].unique()]


def print_target_counts(df: pd.DataFrame, title: str) -> pd.Series:
    """Print and return target class counts."""
    print_section(title)
    counts = df[TARGET_COLUMN].value_counts()
    print(counts)
    return counts


def evaluate_model(model_name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Evaluate a model using the assignment metrics."""
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
    }


def format_confusion_matrix_explanation(
    matrix: np.ndarray,
    class_order: list[str],
) -> list[str]:
    """Create readable confusion matrix lines using the actual class order."""
    dropout_index = class_order.index("Dropout")
    graduate_index = class_order.index("Graduate")
    return [
        f"Class order: {class_order}",
        f"Actual Dropout predicted Dropout = {matrix[dropout_index, dropout_index]}",
        f"Actual Dropout predicted Graduate = {matrix[dropout_index, graduate_index]}",
        f"Actual Graduate predicted Dropout = {matrix[graduate_index, dropout_index]}",
        f"Actual Graduate predicted Graduate = {matrix[graduate_index, graduate_index]}",
    ]


# =============================================================================
# Dataset loading
# =============================================================================


def load_dataset() -> pd.DataFrame:
    """Load the UCI CSV from the URL, falling back to local data.csv if needed."""
    print_section("Loading Dataset")
    try:
        print(f"Attempting to load dataset from UCI URL:\n{UCI_DATA_URL}")
        df = pd.read_csv(UCI_DATA_URL, sep=";")
        print("Dataset loaded successfully from the UCI online CSV URL.")
    except Exception as error:
        print(f"Online loading failed with error: {error}")
        print(f"Attempting local fallback file: {LOCAL_DATA_PATH}")
        if not LOCAL_DATA_PATH.exists():
            raise FileNotFoundError(
                "The online dataset could not be loaded, and data.csv was not "
                f"found at: {LOCAL_DATA_PATH}"
            ) from error
        df = pd.read_csv(LOCAL_DATA_PATH, sep=";")
        print("Dataset loaded successfully from local data.csv.")

    print(
        f"Original dataset contains {df.shape[0]} records and {df.shape[1]} columns. "
        f"After removing the target column, there are {df.shape[1] - 1} input features."
    )
    return df


def validate_required_columns(df: pd.DataFrame) -> None:
    """Stop early if the dataset does not contain the required assignment columns."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise KeyError(
            "The dataset is missing these required columns:\n"
            + "\n".join(f"- {column}" for column in missing_columns)
        )


# =============================================================================
# Data inspection
# =============================================================================


def inspect_dataset(df: pd.DataFrame) -> dict[str, object]:
    """Print the required dataset inspection evidence."""
    print_section("Initial Dataset Inspection")
    print("Dataset shape:")
    print(df.shape)

    print("\nFirst five rows using df.head():")
    print(df.head())

    print("\nDataset information using df.info():")
    df.info()

    print("\nDescriptive statistics using df.describe():")
    print(df.describe())

    descriptive_values = {
        "Admission grade minimum": df["Admission grade"].min(),
        "Admission grade maximum": df["Admission grade"].max(),
        "Admission grade mean": df["Admission grade"].mean(),
        "Age at enrollment minimum": df["Age at enrollment"].min(),
        "Age at enrollment maximum": df["Age at enrollment"].max(),
        "Age at enrollment mean": df["Age at enrollment"].mean(),
    }

    print_section("Descriptive Statistics Values for Report")
    for label, value in descriptive_values.items():
        print(f"{label}: {value}")

    print_section("Missing Value Check")
    missing_by_column = df.isnull().sum()
    missing_total = int(missing_by_column.sum())
    print("Missing values per column:")
    print(missing_by_column)
    print(f"\nMissing values total: {missing_total}")

    print_section("Duplicate Row Check")
    duplicate_count_before = int(df.duplicated().sum())
    print(f"Duplicate rows before removing duplicates: {duplicate_count_before}")

    df_no_duplicates = df.drop_duplicates().reset_index(drop=True)
    duplicate_count_after = int(df_no_duplicates.duplicated().sum())
    print(f"Duplicate rows after removing duplicates: {duplicate_count_after}")
    print(f"Dataset shape after removing duplicates: {df_no_duplicates.shape}")

    return {
        "df_no_duplicates": df_no_duplicates,
        "descriptive_values": descriptive_values,
        "missing_total": missing_total,
        "duplicate_count_before": duplicate_count_before,
        "duplicate_count_after": duplicate_count_after,
    }


def filter_binary_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Remove Enrolled records to keep the task as Dropout vs Graduate."""
    print_section("Filtering to Binary Classification")
    df_binary = df[df[TARGET_COLUMN].isin(BINARY_TARGET_CLASSES)].copy()
    df_binary = df_binary.reset_index(drop=True)

    target_counts = df_binary[TARGET_COLUMN].value_counts()
    class_percentages = target_counts / len(df_binary) * 100
    dropout_count = int(target_counts.get("Dropout", 0))
    graduate_count = int(target_counts.get("Graduate", 0))

    print('Filtered out "Enrolled" records.')
    print(f"Binary classification dataset shape: {df_binary.shape}")
    print(
        "After filtering Enrolled, the working dataset contains "
        f"{len(df_binary)} records: {dropout_count} Dropout and "
        f"{graduate_count} Graduate."
    )
    print(f"Dropout percentage: {class_percentages.get('Dropout', 0):.2f}%")
    print(f"Graduate percentage: {class_percentages.get('Graduate', 0):.2f}%")

    return df_binary, class_percentages


# =============================================================================
# EDA
# =============================================================================


def plot_target_distribution(df: pd.DataFrame, title: str, filename: str, color: str) -> None:
    """Create a target distribution countplot."""
    print_section(f"EDA Plot: {title}")
    if "Enrolled" in df[TARGET_COLUMN].unique():
        target_order = ["Dropout", "Enrolled", "Graduate"]
    else:
        target_order = ordered_targets(df)

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=TARGET_COLUMN, order=target_order, color=color)
    plt.title(title)
    plt.xlabel("Target")
    plt.ylabel("Number of Students")
    save_plot(filename)


def plot_correlation_heatmap(df_binary: pd.DataFrame) -> None:
    """Plot the correlation heatmap for key academic variables."""
    print_section("EDA Plot: Correlation Heatmap")
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        df_binary[ACADEMIC_COLUMNS_FOR_CORRELATION].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Heatmap of Key Academic Variables")
    save_plot("correlation_heatmap.png")


def plot_regression_admission_grade_first_sem_grade(df_binary: pd.DataFrame) -> None:
    """Plot Admission grade vs first semester grade with a regression line."""
    print_section("EDA Plot: Admission Grade vs First Semester Grade")
    plt.figure(figsize=(8, 5))
    sns.regplot(
        data=df_binary,
        x="Admission grade",
        y="Curricular units 1st sem (grade)",
        scatter_kws={"alpha": 0.45},
        line_kws={"color": "red"},
    )
    plt.title("Admission Grade vs Curricular Units 1st Sem Grade")
    plt.xlabel("Admission grade")
    plt.ylabel("Curricular units 1st sem (grade)")
    save_plot("regression_admission_grade_first_sem_grade.png")


def plot_boxplot_by_target(
    df_binary: pd.DataFrame,
    y_column: str,
    title: str,
    ylabel: str,
    filename: str,
) -> None:
    """Create a boxplot for a numeric variable grouped by Target."""
    print_section(f"EDA Plot: {title}")
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df_binary, x=TARGET_COLUMN, y=y_column, order=ordered_targets(df_binary))
    plt.title(title)
    plt.xlabel("Target")
    plt.ylabel(ylabel)
    save_plot(filename)


def plot_countplot_by_target(
    df_binary: pd.DataFrame,
    x_column: str,
    title: str,
    xlabel: str,
    filename: str,
) -> None:
    """Create a countplot for a categorical/binary variable grouped by Target."""
    print_section(f"EDA Plot: {title}")
    plt.figure(figsize=(8, 5))
    sns.countplot(
        data=df_binary,
        x=x_column,
        hue=TARGET_COLUMN,
        hue_order=ordered_targets(df_binary),
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of Students")
    plt.legend(title="Target")
    save_plot(filename)


def run_eda(df: pd.DataFrame, df_binary: pd.DataFrame) -> tuple[float, float, pd.DataFrame]:
    """Run all required EDA plots and statistical summaries."""
    plot_target_distribution(
        df,
        "Target Distribution Before Filtering",
        "target_distribution_before_filtering.png",
        "#4C72B0",
    )
    plot_target_distribution(
        df_binary,
        "Target Distribution After Filtering",
        "target_distribution_after_filtering.png",
        "#55A868",
    )
    plot_correlation_heatmap(df_binary)
    plot_regression_admission_grade_first_sem_grade(df_binary)
    plot_boxplot_by_target(
        df_binary,
        "Age at enrollment",
        "Age at Enrollment by Target",
        "Age at enrollment",
        "age_target_boxplot.png",
    )
    plot_boxplot_by_target(
        df_binary,
        "Admission grade",
        "Admission Grade by Target",
        "Admission grade",
        "admission_grade_target_boxplot.png",
    )
    plot_countplot_by_target(
        df_binary,
        "Tuition fees up to date",
        "Tuition Fees Up to Date by Target",
        "Tuition fees up to date (0 = No, 1 = Yes)",
        "tuition_fees_target_countplot.png",
    )
    plot_countplot_by_target(
        df_binary,
        "Scholarship holder",
        "Scholarship Holder by Target",
        "Scholarship holder (0 = No, 1 = Yes)",
        "scholarship_target_countplot.png",
    )
    plot_boxplot_by_target(
        df_binary,
        "Curricular units 1st sem (approved)",
        "First Semester Approved Units by Target",
        "Curricular units 1st sem (approved)",
        "first_sem_approved_units_boxplot.png",
    )
    plot_boxplot_by_target(
        df_binary,
        "Curricular units 2nd sem (approved)",
        "Second Semester Approved Units by Target",
        "Curricular units 2nd sem (approved)",
        "second_sem_approved_units_boxplot.png",
    )

    print_section("Pearson Correlation Analysis")
    first_sem_column = "Curricular units 1st sem (approved)"
    second_sem_column = "Curricular units 2nd sem (approved)"
    pearson_r, pearson_p_value = pearsonr(
        df_binary[first_sem_column],
        df_binary[second_sem_column],
    )
    print(f"Pearson correlation between {first_sem_column} and {second_sem_column}:")
    print(f"Pearson r: {pearson_r:.4f}")
    print(f"p-value: {pearson_p_value:.6g}")

    print_section("Median Values by Target")
    median_table = df_binary.groupby(TARGET_COLUMN)[MEDIAN_COLUMNS].median()
    print(median_table)

    return float(pearson_r), float(pearson_p_value), median_table


# =============================================================================
# Preprocessing
# =============================================================================


def encode_target(df_binary: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder, dict[str, int]]:
    """Encode Dropout/Graduate target labels using LabelEncoder."""
    print_section("Target Encoding")
    df_model = df_binary.copy()
    target_encoder = LabelEncoder()
    df_model["Target_encoded"] = target_encoder.fit_transform(df_model[TARGET_COLUMN])

    target_mapping = {
        class_name: int(encoded_value)
        for class_name, encoded_value in zip(
            target_encoder.classes_,
            target_encoder.transform(target_encoder.classes_),
        )
    }

    print("Target mapping created by LabelEncoder:")
    for class_name, encoded_value in target_mapping.items():
        print(f"{class_name} -> {encoded_value}")
    print(
        "LabelEncoder mapping is alphabetical. In this run: Dropout = 0 and "
        "Graduate = 1. Therefore, sklearn's default positive class for "
        "precision/recall/F1 is Graduate, not Dropout."
    )

    return df_model, target_encoder, target_mapping


def analyze_age_outliers(df_binary: pd.DataFrame) -> int:
    """Perform IQR outlier analysis for Age at enrollment."""
    print_section("IQR Outlier Analysis: Age at Enrollment")
    age_column = "Age at enrollment"
    q1 = df_binary[age_column].quantile(0.25)
    q3 = df_binary[age_column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = int(
        ((df_binary[age_column] < lower_bound) | (df_binary[age_column] > upper_bound)).sum()
    )

    print(f"Q1: {q1:.2f}")
    print(f"Q3: {q3:.2f}")
    print(f"IQR: {iqr:.2f}")
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")
    print(f"Number of outlier records: {outlier_count}")
    return outlier_count


def create_features_and_target(
    df_model: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Separate model features from encoded target."""
    print_section("Separating Features and Target")
    X = df_model.drop(columns=[TARGET_COLUMN, "Target_encoded"])
    y = df_model["Target_encoded"]

    non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_columns:
        print("Non-numeric feature columns found and one-hot encoded:")
        print(non_numeric_columns)
        X = pd.get_dummies(X, columns=non_numeric_columns, drop_first=True)
    else:
        print("All feature columns are numeric.")

    feature_columns = X.columns.tolist()
    print(f"Number of feature columns: {len(feature_columns)}")
    print(f"Target vector length: {len(y)}")
    return X, y, feature_columns


def split_and_scale_features(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler, pd.DataFrame, pd.DataFrame]:
    """Split data, scale features, and print before/after scaling previews."""
    print_section("Train-Test Split")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train class counts:\n{y_train.value_counts().sort_index()}")
    print(f"y_test class counts:\n{y_test.value_counts().sort_index()}")

    print_section("Feature Scaling")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X.columns,
        index=X_test.index,
    )

    preview_columns = [column for column in SCALING_PREVIEW_COLUMNS if column in X.columns]
    scaling_before = X_train[preview_columns].head()
    scaling_after = X_train_scaled[preview_columns].head()
    print("Before scaling preview from X_train:")
    print(scaling_before)
    print("\nAfter scaling preview from X_train:")
    print(scaling_after)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, scaling_before, scaling_after


# =============================================================================
# Model building
# =============================================================================


def train_models(
    X_train_scaled: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[dict[str, object], dict[str, object]]:
    """Train the required models and tune Random Forest with GridSearchCV."""
    print_section("Model Training")

    logistic_regression = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    logistic_regression.fit(X_train_scaled, y_train)
    print("Trained Logistic Regression.")

    basic_random_forest = RandomForestClassifier(random_state=RANDOM_STATE)
    basic_random_forest.fit(X_train_scaled, y_train)
    print("Trained Basic Random Forest Classifier.")

    gradient_boosting = GradientBoostingClassifier(random_state=RANDOM_STATE)
    gradient_boosting.fit(X_train_scaled, y_train)
    print("Trained Gradient Boosting Classifier.")

    parameter_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid=parameter_grid,
        cv=3,
        scoring="recall",
        n_jobs=-1,
    )
    grid_search.fit(X_train_scaled, y_train)

    print("\nBest Random Forest parameters from GridSearchCV:")
    print(grid_search.best_params_)

    models = {
        "Logistic Regression": logistic_regression,
        "Basic Random Forest": basic_random_forest,
        "Gradient Boosting": gradient_boosting,
        "Tuned Random Forest": grid_search.best_estimator_,
    }
    return models, grid_search.best_params_


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_models(
    models: dict[str, object],
    X_test_scaled: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Evaluate all trained models and save model_results.csv."""
    print_section("Model Evaluation")
    predictions = {}
    results = []

    for model_name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        predictions[model_name] = y_pred
        results.append(evaluate_model(model_name, y_test, y_pred))

    results_df = pd.DataFrame(results)
    print("Model comparison table:")
    print(results_df.to_string(index=False))

    output_path = PROJECT_DIR / "model_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved model comparison table to: {output_path}")

    return results_df, predictions


def evaluate_tuned_random_forest(
    y_test: pd.Series,
    tuned_rf_predictions: np.ndarray,
    target_encoder: LabelEncoder,
) -> tuple[str, dict[str, dict[str, float]], dict[str, float], np.ndarray, list[str]]:
    """Print tuned RF classification report, dropout metrics, and confusion matrix."""
    class_order = list(target_encoder.classes_)

    print_section("Classification Report: Tuned Random Forest")
    report_text = classification_report(
        y_test,
        tuned_rf_predictions,
        target_names=class_order,
        zero_division=0,
    )
    report_dict = classification_report(
        y_test,
        tuned_rf_predictions,
        target_names=class_order,
        output_dict=True,
        zero_division=0,
    )
    print(report_text)

    dropout_metrics = {
        "precision": report_dict["Dropout"]["precision"],
        "recall": report_dict["Dropout"]["recall"],
        "f1-score": report_dict["Dropout"]["f1-score"],
    }
    print_section("Dropout-Focused Metrics: Tuned Random Forest")
    print(f"Dropout precision: {dropout_metrics['precision']:.6f}")
    print(f"Dropout recall: {dropout_metrics['recall']:.6f}")
    print(f"Dropout F1-score: {dropout_metrics['f1-score']:.6f}")

    print_section("Confusion Matrix: Tuned Random Forest")
    encoded_labels = target_encoder.transform(class_order)
    matrix = confusion_matrix(y_test, tuned_rf_predictions, labels=encoded_labels)
    explanation_lines = format_confusion_matrix_explanation(matrix, class_order)
    print("Confusion matrix explanation:")
    for line in explanation_lines:
        print(line)

    return report_text, report_dict, dropout_metrics, matrix, class_order


def plot_model_comparison_bar_chart(results_df: pd.DataFrame) -> None:
    """Create the model comparison bar chart."""
    print_section("Plot: Model Comparison Bar Chart")
    results_long = results_df.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1-score"],
        var_name="Metric",
        value_name="Score",
    )

    plt.figure(figsize=(11, 6))
    sns.barplot(data=results_long, x="Model", y="Score", hue="Metric")
    plt.title("Model Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="Metric", loc="lower right")
    save_plot("model_comparison_bar_chart.png")


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_order: list[str],
) -> None:
    """Save the tuned Random Forest confusion matrix plot."""
    print_section("Plot: Confusion Matrix for Tuned Random Forest")
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_order,
        yticklabels=class_order,
    )
    plt.title("Confusion Matrix: Tuned Random Forest")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    save_plot("confusion_matrix_tuned_random_forest.png")


# =============================================================================
# Feature importance
# =============================================================================


def create_feature_importance(
    tuned_random_forest: RandomForestClassifier,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Print and plot the top tuned Random Forest feature importances."""
    print_section("Feature Importance: Tuned Random Forest")
    feature_importance_df = pd.DataFrame(
        {
            "Feature": feature_columns,
            "Importance": tuned_random_forest.feature_importances_,
        }
    ).sort_values(by="Importance", ascending=False)

    top_10 = feature_importance_df.head(10)
    print("Top 10 feature importances:")
    print(top_10.to_string(index=False))

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=top_10.sort_values(by="Importance", ascending=True),
        x="Importance",
        y="Feature",
        color="#4C72B0",
    )
    plt.title("Top 10 Feature Importances: Tuned Random Forest")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    save_plot("feature_importance_tuned_random_forest.png")

    return feature_importance_df


# =============================================================================
# Saving outputs
# =============================================================================


def save_streamlit_artifacts(
    tuned_random_forest: RandomForestClassifier,
    scaler: StandardScaler,
    target_encoder: LabelEncoder,
    feature_columns: list[str],
) -> None:
    """Save files required by the Streamlit app."""
    print_section("Saving Files for Streamlit App")
    artifacts = {
        "student_dropout_model.pkl": tuned_random_forest,
        "scaler.pkl": scaler,
        "target_encoder.pkl": target_encoder,
        "feature_columns.pkl": feature_columns,
    }
    for filename, artifact in artifacts.items():
        output_path = PROJECT_DIR / filename
        joblib.dump(artifact, output_path)
        print(f"Saved {filename} to: {output_path}")


def save_report_values(report_values: dict[str, object]) -> None:
    """Save the terminal report evidence to report_values.txt."""
    report_path = PROJECT_DIR / "report_values.txt"
    matrix = report_values["confusion_matrix"]
    class_order = report_values["class_order"]
    explanation_lines = format_confusion_matrix_explanation(matrix, class_order)

    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("COMP 30044 CW2 Report Values\n")
        report_file.write("=" * 80 + "\n\n")

        report_file.write("Dataset shape\n")
        report_file.write(f"{report_values['dataset_shape']}\n\n")

        report_file.write("Missing values total\n")
        report_file.write(f"{report_values['missing_total']}\n\n")

        report_file.write("Duplicate rows\n")
        report_file.write(
            f"Before removal: {report_values['duplicate_count_before']}\n"
        )
        report_file.write(
            f"After removal: {report_values['duplicate_count_after']}\n\n"
        )

        report_file.write("Target counts before filtering\n")
        report_file.write(report_values["target_counts_before"].to_string())
        report_file.write("\n\n")

        report_file.write("Target counts after filtering\n")
        report_file.write(report_values["target_counts_after"].to_string())
        report_file.write("\n\n")

        report_file.write("Class percentages after filtering\n")
        for target_class in BINARY_TARGET_CLASSES:
            percentage = report_values["class_percentages"].get(target_class, 0)
            report_file.write(f"{target_class}: {percentage:.2f}%\n")
        report_file.write("\n")

        report_file.write("Descriptive statistics values for report\n")
        for label, value in report_values["descriptive_values"].items():
            report_file.write(f"{label}: {value}\n")
        report_file.write("\n")

        report_file.write("Pearson correlation\n")
        report_file.write(f"Pearson r: {report_values['pearson_r']}\n")
        report_file.write(f"p-value: {report_values['pearson_p_value']}\n\n")

        report_file.write("Medians by Target\n")
        report_file.write(report_values["median_table"].to_string())
        report_file.write("\n\n")

        report_file.write("IQR outlier count for Age at enrollment\n")
        report_file.write(f"{report_values['iqr_outlier_count']}\n\n")

        report_file.write("Target encoding mapping\n")
        report_file.write(f"{report_values['target_mapping']}\n\n")

        report_file.write("Train/test shapes\n")
        report_file.write(f"X_train: {report_values['x_train_shape']}\n")
        report_file.write(f"X_test: {report_values['x_test_shape']}\n\n")

        report_file.write("Scaling preview before\n")
        report_file.write(report_values["scaling_before"].to_string())
        report_file.write("\n\n")

        report_file.write("Scaling preview after\n")
        report_file.write(report_values["scaling_after"].to_string())
        report_file.write("\n\n")

        report_file.write("Best Random Forest parameters\n")
        report_file.write(f"{report_values['best_random_forest_params']}\n\n")

        report_file.write("Model comparison table\n")
        report_file.write(report_values["results_df"].to_string(index=False))
        report_file.write("\n\n")

        report_file.write("Class order\n")
        report_file.write(f"{class_order}\n\n")

        report_file.write("Confusion matrix\n")
        report_file.write(np.array2string(matrix))
        report_file.write("\n\n")

        report_file.write("Confusion matrix explanation\n")
        for line in explanation_lines:
            report_file.write(f"{line}\n")
        report_file.write("\n")

        report_file.write("Classification report\n")
        report_file.write(report_values["classification_report_text"])
        report_file.write("\n")

        report_file.write("Dropout-specific metrics\n")
        for metric_name, metric_value in report_values["dropout_metrics"].items():
            report_file.write(f"Dropout {metric_name}: {metric_value}\n")
        report_file.write("\n")

        report_file.write("Top 10 feature importances\n")
        report_file.write(report_values["feature_importance_df"].head(10).to_string(index=False))
        report_file.write("\n")

    print(f"Saved report values to: {report_path}")


def main() -> None:
    """Run the full assignment workflow."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = load_dataset()
    validate_required_columns(df)
    inspection = inspect_dataset(df)
    df_clean = inspection["df_no_duplicates"]

    target_counts_before = print_target_counts(df_clean, "Target Class Counts Before Filtering")
    df_binary, class_percentages = filter_binary_dataset(df_clean)
    target_counts_after = print_target_counts(df_binary, "Target Class Counts After Filtering")

    pearson_r, pearson_p_value, median_table = run_eda(df_clean, df_binary)
    df_model, target_encoder, target_mapping = encode_target(df_binary)
    iqr_outlier_count = analyze_age_outliers(df_binary)

    X, y, feature_columns = create_features_and_target(df_model)
    (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        scaling_before,
        scaling_after,
    ) = split_and_scale_features(X, y)

    models, best_random_forest_params = train_models(X_train_scaled, y_train)
    results_df, predictions = evaluate_models(models, X_test_scaled, y_test)

    tuned_random_forest = models["Tuned Random Forest"]
    tuned_rf_predictions = predictions["Tuned Random Forest"]
    (
        classification_report_text,
        _classification_report_dict,
        dropout_metrics,
        confusion_matrix_values,
        class_order,
    ) = evaluate_tuned_random_forest(y_test, tuned_rf_predictions, target_encoder)

    plot_model_comparison_bar_chart(results_df)
    plot_confusion_matrix(confusion_matrix_values, class_order)
    feature_importance_df = create_feature_importance(tuned_random_forest, feature_columns)

    save_streamlit_artifacts(tuned_random_forest, scaler, target_encoder, feature_columns)
    save_report_values(
        {
            "dataset_shape": df_clean.shape,
            "missing_total": inspection["missing_total"],
            "duplicate_count_before": inspection["duplicate_count_before"],
            "duplicate_count_after": inspection["duplicate_count_after"],
            "target_counts_before": target_counts_before,
            "target_counts_after": target_counts_after,
            "class_percentages": class_percentages,
            "descriptive_values": inspection["descriptive_values"],
            "pearson_r": pearson_r,
            "pearson_p_value": pearson_p_value,
            "median_table": median_table,
            "iqr_outlier_count": iqr_outlier_count,
            "target_mapping": target_mapping,
            "x_train_shape": X_train_scaled.shape,
            "x_test_shape": X_test_scaled.shape,
            "scaling_before": scaling_before,
            "scaling_after": scaling_after,
            "best_random_forest_params": best_random_forest_params,
            "results_df": results_df,
            "class_order": class_order,
            "confusion_matrix": confusion_matrix_values,
            "classification_report_text": classification_report_text,
            "dropout_metrics": dropout_metrics,
            "feature_importance_df": feature_importance_df,
        }
    )

    print_section("Workflow Complete")
    print("Binary classification workflow completed: Dropout vs Graduate.")
    print("All plots, model results, report values, and Streamlit app files have been saved.")


if __name__ == "__main__":
    main()

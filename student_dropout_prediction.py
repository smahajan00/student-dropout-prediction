"""
COMP 30044 CW2
Predicting Student Dropout and Academic Success using Machine Learning

This script loads the UCI student dropout dataset, performs exploratory data
analysis, trains several binary classification models, evaluates them, and
saves the best tuned Random Forest model for use in a Streamlit app.
"""

from pathlib import Path
import os
import warnings

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib_cache"))

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
        + [
            "Tuition fees up to date",
            "Scholarship holder",
        ]
    )
)


def print_section(title: str) -> None:
    """Print a clear heading so console output is easy to screenshot."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_original_dataset_note(df: pd.DataFrame) -> None:
    """Print a concise note about records, columns, and feature count."""
    feature_count = df.shape[1] - 1
    print(
        f"Original dataset contains {df.shape[0]} records and {df.shape[1]} columns. "
        f"After removing the target column, there are {feature_count} input features."
    )


def save_plot(filename: str) -> None:
    """Save the current Matplotlib figure into the figures folder, then show it."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    plt.show()
    plt.close()


def load_dataset() -> pd.DataFrame:
    """
    Load the UCI dataset from the online URL.

    If the URL is unavailable, fall back to a local file named data.csv in the
    same project folder. The dataset uses semicolons as separators.
    """
    print_section("Loading Dataset")
    try:
        print(f"Attempting to load dataset from UCI URL:\n{UCI_DATA_URL}")
        df = pd.read_csv(UCI_DATA_URL, sep=";")
        print("Dataset loaded successfully from the UCI online CSV URL.")
        return df
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
        return df


def validate_required_columns(df: pd.DataFrame) -> None:
    """Fail early with a helpful message if expected dataset columns are missing."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise KeyError(
            "The dataset is missing the following required columns:\n"
            + "\n".join(f"- {column}" for column in missing_columns)
        )


def print_initial_dataset_summary(df: pd.DataFrame) -> None:
    """Print the required initial inspection outputs."""
    print_section("Initial Dataset Inspection")
    print("Dataset shape:")
    print(df.shape)

    print("\nFirst five rows using df.head():")
    print(df.head())

    print("\nDataset information using df.info():")
    df.info()

    print("\nDescriptive statistics using df.describe():")
    print(df.describe())

    print_section("Missing Value Check")
    print("Missing values per column:")
    print(df.isnull().sum())


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Print duplicate counts before and after removing duplicates."""
    print_section("Duplicate Row Check")
    duplicate_count_before = df.duplicated().sum()
    print(f"Duplicate rows before removing duplicates: {duplicate_count_before}")

    df_no_duplicates = df.drop_duplicates().reset_index(drop=True)

    duplicate_count_after = df_no_duplicates.duplicated().sum()
    print(f"Duplicate rows after removing duplicates: {duplicate_count_after}")
    print(f"Dataset shape after removing duplicates: {df_no_duplicates.shape}")

    return df_no_duplicates


def print_target_counts(df: pd.DataFrame, heading: str) -> None:
    """Print target class counts in a clear format."""
    print_section(heading)
    print(df[TARGET_COLUMN].value_counts())


def plot_target_distribution_before_filtering(df: pd.DataFrame) -> None:
    """Plot the full multiclass target distribution before binary filtering."""
    print_section("EDA Plot: Target Distribution Before Filtering")
    target_order = [
        target
        for target in ["Dropout", "Enrolled", "Graduate"]
        if target in df[TARGET_COLUMN].unique()
    ]

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=TARGET_COLUMN, order=target_order, color="#4C72B0")
    plt.title("Target Distribution Before Filtering")
    plt.xlabel("Target")
    plt.ylabel("Number of Students")
    save_plot("target_distribution_before_filtering.png")


def filter_to_binary_classification(df: pd.DataFrame) -> pd.DataFrame:
    """Remove Enrolled records so the task becomes Dropout vs Graduate."""
    print_section("Filtering to Binary Classification")
    df_binary = df[df[TARGET_COLUMN].isin(BINARY_TARGET_CLASSES)].copy()
    df_binary = df_binary.reset_index(drop=True)
    target_counts = df_binary[TARGET_COLUMN].value_counts()
    dropout_count = int(target_counts.get("Dropout", 0))
    graduate_count = int(target_counts.get("Graduate", 0))
    class_percentages = df_binary[TARGET_COLUMN].value_counts(normalize=True) * 100

    print('Filtered out "Enrolled" records.')
    print(f"Binary classification dataset shape: {df_binary.shape}")
    print(
        "After filtering Enrolled, the working dataset contains "
        f"{len(df_binary)} records: {dropout_count} Dropout and "
        f"{graduate_count} Graduate."
    )
    print(f"Dropout percentage: {class_percentages.get('Dropout', 0):.2f}%")
    print(f"Graduate percentage: {class_percentages.get('Graduate', 0):.2f}%")
    return df_binary


def plot_target_distribution_after_filtering(df_binary: pd.DataFrame) -> None:
    """Plot the binary target distribution after removing Enrolled records."""
    print_section("EDA Plot: Target Distribution After Filtering")
    target_order = [
        target for target in BINARY_TARGET_CLASSES if target in df_binary[TARGET_COLUMN].unique()
    ]

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df_binary, x=TARGET_COLUMN, order=target_order, color="#55A868")
    plt.title("Target Distribution After Filtering")
    plt.xlabel("Target")
    plt.ylabel("Number of Students")
    save_plot("target_distribution_after_filtering.png")


def plot_correlation_heatmap(df_binary: pd.DataFrame) -> None:
    """Plot a correlation heatmap for key academic variables."""
    print_section("EDA Plot: Correlation Heatmap")
    correlation_matrix = df_binary[ACADEMIC_COLUMNS_FOR_CORRELATION].corr()

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        correlation_matrix,
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
    """Plot Admission grade against first semester grade with a regression line."""
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
    """Reusable helper for boxplots grouped by the binary target variable."""
    target_order = [
        target for target in BINARY_TARGET_CLASSES if target in df_binary[TARGET_COLUMN].unique()
    ]

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df_binary, x=TARGET_COLUMN, y=y_column, order=target_order)
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
    """Reusable helper for countplots with Target as the hue."""
    target_order = [
        target for target in BINARY_TARGET_CLASSES if target in df_binary[TARGET_COLUMN].unique()
    ]

    plt.figure(figsize=(8, 5))
    sns.countplot(
        data=df_binary,
        x=x_column,
        hue=TARGET_COLUMN,
        hue_order=target_order,
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of Students")
    plt.legend(title="Target")
    save_plot(filename)


def run_eda_plots(df_binary: pd.DataFrame) -> None:
    """Create and save the required exploratory data analysis plots."""
    plot_correlation_heatmap(df_binary)
    plot_regression_admission_grade_first_sem_grade(df_binary)

    print_section("EDA Plot: Age at Enrollment vs Target")
    plot_boxplot_by_target(
        df_binary,
        y_column="Age at enrollment",
        title="Age at Enrollment by Target",
        ylabel="Age at enrollment",
        filename="age_target_boxplot.png",
    )

    print_section("EDA Plot: Admission Grade vs Target")
    plot_boxplot_by_target(
        df_binary,
        y_column="Admission grade",
        title="Admission Grade by Target",
        ylabel="Admission grade",
        filename="admission_grade_target_boxplot.png",
    )

    print_section("EDA Plot: Tuition Fees Up to Date vs Target")
    plot_countplot_by_target(
        df_binary,
        x_column="Tuition fees up to date",
        title="Tuition Fees Up to Date by Target",
        xlabel="Tuition fees up to date (0 = No, 1 = Yes)",
        filename="tuition_fees_target_countplot.png",
    )

    print_section("EDA Plot: Scholarship Holder vs Target")
    plot_countplot_by_target(
        df_binary,
        x_column="Scholarship holder",
        title="Scholarship Holder by Target",
        xlabel="Scholarship holder (0 = No, 1 = Yes)",
        filename="scholarship_target_countplot.png",
    )

    print_section("EDA Plot: First Semester Approved Units vs Target")
    plot_boxplot_by_target(
        df_binary,
        y_column="Curricular units 1st sem (approved)",
        title="First Semester Approved Units by Target",
        ylabel="Curricular units 1st sem (approved)",
        filename="first_sem_approved_units_boxplot.png",
    )

    print_section("EDA Plot: Second Semester Approved Units vs Target")
    plot_boxplot_by_target(
        df_binary,
        y_column="Curricular units 2nd sem (approved)",
        title="Second Semester Approved Units by Target",
        ylabel="Curricular units 2nd sem (approved)",
        filename="second_sem_approved_units_boxplot.png",
    )


def print_pearson_correlation(df_binary: pd.DataFrame) -> tuple[float, float]:
    """Calculate and print Pearson correlation for approved units across semesters."""
    print_section("Pearson Correlation Analysis")
    first_sem_column = "Curricular units 1st sem (approved)"
    second_sem_column = "Curricular units 2nd sem (approved)"

    correlation_data = df_binary[[first_sem_column, second_sem_column]].dropna()
    pearson_r, p_value = pearsonr(
        correlation_data[first_sem_column],
        correlation_data[second_sem_column],
    )

    print(
        "Pearson correlation between "
        f"{first_sem_column} and {second_sem_column}:"
    )
    print(f"Pearson r: {pearson_r:.4f}")
    print(f"p-value: {p_value:.6g}")
    return float(pearson_r), float(p_value)


def print_median_values_by_target(df_binary: pd.DataFrame) -> pd.DataFrame:
    """Print median values by Target for the required academic variables."""
    print_section("Median Values by Target")
    median_table = df_binary.groupby(TARGET_COLUMN)[MEDIAN_COLUMNS].median()
    print(median_table)
    return median_table


def encode_target_variable(df_binary: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """Encode Target with LabelEncoder and print the mapping clearly."""
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

    positive_class_name = target_encoder.inverse_transform([1])[0]
    print(
        "For binary precision, recall, and F1-score, sklearn's default "
        f"positive encoded class is 1: {positive_class_name}"
    )
    print(
        "LabelEncoder mapping is alphabetical. In this run: Dropout = 0 and "
        "Graduate = 1. Therefore, sklearn's default positive class for "
        "precision/recall/F1 is Graduate, not Dropout."
    )

    return df_model, target_encoder


def print_iqr_outlier_analysis(df_binary: pd.DataFrame) -> int:
    """Perform IQR outlier analysis for Age at enrollment."""
    print_section("IQR Outlier Analysis: Age at Enrollment")
    age_column = "Age at enrollment"

    q1 = df_binary[age_column].quantile(0.25)
    q3 = df_binary[age_column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (df_binary[age_column] < lower_bound) | (
        df_binary[age_column] > upper_bound
    )
    outlier_count = int(outlier_mask.sum())

    print(f"Q1: {q1:.2f}")
    print(f"Q3: {q3:.2f}")
    print(f"IQR: {iqr:.2f}")
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")
    print(f"Number of outlier records: {outlier_count}")
    return outlier_count


def separate_features_and_target(
    df_model: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Separate feature columns from the encoded target."""
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """Train-test split, apply StandardScaler, and print scaling previews."""
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
    X_train_scaled_array = scaler.fit_transform(X_train)
    X_test_scaled_array = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled_array,
        columns=X.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled_array,
        columns=X.columns,
        index=X_test.index,
    )

    preview_columns = [column for column in SCALING_PREVIEW_COLUMNS if column in X.columns]
    print("Before scaling preview from X_train:")
    print(X_train[preview_columns].head())

    print("\nAfter scaling preview from X_train:")
    print(X_train_scaled[preview_columns].head())

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_models(
    X_train_scaled: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[dict[str, object], dict[str, object]]:
    """Train Logistic Regression, Random Forest, Gradient Boosting, and tuned RF."""
    print_section("Model Training")

    logistic_regression = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
    )
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


def evaluate_models(
    models: dict[str, object],
    X_test_scaled: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Evaluate all models using Accuracy, Precision, Recall, and F1-score."""
    print_section("Model Evaluation")
    results = []
    predictions = {}

    for model_name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        predictions[model_name] = y_pred

        results.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1-score": f1_score(y_test, y_pred, zero_division=0),
            }
        )

    results_df = pd.DataFrame(results)

    print("Model comparison table:")
    print(results_df.to_string(index=False))

    results_output_path = PROJECT_DIR / "model_results.csv"
    results_df.to_csv(results_output_path, index=False)
    print(f"\nSaved model comparison table to: {results_output_path}")

    return results_df, predictions


def print_tuned_random_forest_report(
    y_test: pd.Series,
    tuned_rf_predictions: np.ndarray,
    target_encoder: LabelEncoder,
) -> tuple[str, dict[str, dict[str, float]], dict[str, float]]:
    """Print the classification report for the tuned Random Forest."""
    print_section("Classification Report: Tuned Random Forest")
    report_text = classification_report(
        y_test,
        tuned_rf_predictions,
        target_names=target_encoder.classes_,
        zero_division=0,
    )
    report_dict = classification_report(
        y_test,
        tuned_rf_predictions,
        target_names=target_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    dropout_metrics = {
        "precision": report_dict["Dropout"]["precision"],
        "recall": report_dict["Dropout"]["recall"],
        "f1-score": report_dict["Dropout"]["f1-score"],
    }

    print(report_text)

    print_section("Dropout-Focused Metrics: Tuned Random Forest")
    print(f"Dropout precision: {dropout_metrics['precision']:.6f}")
    print(f"Dropout recall: {dropout_metrics['recall']:.6f}")
    print(f"Dropout F1-score: {dropout_metrics['f1-score']:.6f}")

    return report_text, report_dict, dropout_metrics


def plot_model_comparison_bar_chart(results_df: pd.DataFrame) -> None:
    """Create and save a grouped bar chart for model performance metrics."""
    print_section("Plot: Model Comparison Bar Chart")
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-score"]
    results_long = results_df.melt(
        id_vars="Model",
        value_vars=metrics_to_plot,
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


def plot_confusion_matrix_tuned_random_forest(
    y_test: pd.Series,
    tuned_rf_predictions: np.ndarray,
    target_encoder: LabelEncoder,
) -> np.ndarray:
    """Create and save a confusion matrix for the tuned Random Forest."""
    print_section("Plot: Confusion Matrix for Tuned Random Forest")
    class_order = list(target_encoder.classes_)
    encoded_labels = target_encoder.transform(class_order)
    matrix = confusion_matrix(y_test, tuned_rf_predictions, labels=encoded_labels)

    dropout_index = class_order.index("Dropout")
    graduate_index = class_order.index("Graduate")

    print("Confusion matrix explanation:")
    print(f"Class order: {class_order}")
    print(
        "Actual Dropout predicted Dropout = "
        f"{matrix[dropout_index, dropout_index]}"
    )
    print(
        "Actual Dropout predicted Graduate = "
        f"{matrix[dropout_index, graduate_index]}"
    )
    print(
        "Actual Graduate predicted Dropout = "
        f"{matrix[graduate_index, dropout_index]}"
    )
    print(
        "Actual Graduate predicted Graduate = "
        f"{matrix[graduate_index, graduate_index]}"
    )

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
    return matrix


def create_feature_importance(
    tuned_random_forest: RandomForestClassifier,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Print and plot the top 10 tuned Random Forest feature importances."""
    print_section("Feature Importance: Tuned Random Forest")
    feature_importance_df = pd.DataFrame(
        {
            "Feature": feature_columns,
            "Importance": tuned_random_forest.feature_importances_,
        }
    ).sort_values(by="Importance", ascending=False)

    top_10_features = feature_importance_df.head(10)
    print("Top 10 feature importances:")
    print(top_10_features.to_string(index=False))

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=top_10_features.sort_values(by="Importance", ascending=True),
        x="Importance",
        y="Feature",
        color="#4C72B0",
    )
    plt.title("Top 10 Feature Importances: Tuned Random Forest")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    save_plot("feature_importance_tuned_random_forest.png")

    return feature_importance_df


def save_report_values(
    dataset_shape: tuple[int, int],
    target_counts_before: pd.Series,
    target_counts_after: pd.Series,
    class_percentages: pd.Series,
    pearson_r: float,
    pearson_p_value: float,
    median_table: pd.DataFrame,
    iqr_outlier_count: int,
    best_random_forest_params: dict[str, object],
    results_df: pd.DataFrame,
    class_order: list[str],
    confusion_matrix_values: np.ndarray,
    classification_report_text: str,
    feature_importance_df: pd.DataFrame,
) -> None:
    """Save key report values to a plain text file for assignment write-up."""
    report_path = PROJECT_DIR / "report_values.txt"
    top_10_feature_importances = feature_importance_df.head(10)

    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("COMP 30044 CW2 Report Values\n")
        report_file.write("=" * 80 + "\n\n")

        report_file.write("Dataset shape\n")
        report_file.write(f"{dataset_shape}\n\n")

        report_file.write("Target counts before filtering\n")
        report_file.write(target_counts_before.to_string())
        report_file.write("\n\n")

        report_file.write("Target counts after filtering\n")
        report_file.write(target_counts_after.to_string())
        report_file.write("\n\n")

        report_file.write("Class percentages after filtering\n")
        for target_class in BINARY_TARGET_CLASSES:
            report_file.write(
                f"{target_class}: {class_percentages.get(target_class, 0):.2f}%\n"
            )
        report_file.write("\n")

        report_file.write("Pearson correlation\n")
        report_file.write(f"Pearson r: {pearson_r}\n")
        report_file.write(f"p-value: {pearson_p_value}\n\n")

        report_file.write("Medians by Target\n")
        report_file.write(median_table.to_string())
        report_file.write("\n\n")

        report_file.write("IQR outlier count for Age at enrollment\n")
        report_file.write(f"{iqr_outlier_count}\n\n")

        report_file.write("Best Random Forest parameters\n")
        report_file.write(f"{best_random_forest_params}\n\n")

        report_file.write("Model comparison table\n")
        report_file.write(results_df.to_string(index=False))
        report_file.write("\n\n")

        report_file.write("Class order\n")
        report_file.write(f"{class_order}\n\n")

        report_file.write("Confusion matrix\n")
        report_file.write(np.array2string(confusion_matrix_values))
        report_file.write("\n\n")

        report_file.write("Classification report\n")
        report_file.write(classification_report_text)
        report_file.write("\n")

        report_file.write("Top 10 feature importances\n")
        report_file.write(top_10_feature_importances.to_string(index=False))
        report_file.write("\n")

    print(f"Saved report values to: {report_path}")


def save_streamlit_app_files(
    tuned_random_forest: RandomForestClassifier,
    scaler: StandardScaler,
    target_encoder: LabelEncoder,
    feature_columns: list[str],
) -> None:
    """Save model artifacts required by the Streamlit app."""
    print_section("Saving Files for Streamlit App")

    model_path = PROJECT_DIR / "student_dropout_model.pkl"
    scaler_path = PROJECT_DIR / "scaler.pkl"
    encoder_path = PROJECT_DIR / "target_encoder.pkl"
    feature_columns_path = PROJECT_DIR / "feature_columns.pkl"

    joblib.dump(tuned_random_forest, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(target_encoder, encoder_path)
    joblib.dump(feature_columns, feature_columns_path)

    print(f"Saved tuned Random Forest model to: {model_path}")
    print(f"Saved StandardScaler to: {scaler_path}")
    print(f"Saved LabelEncoder to: {encoder_path}")
    print(f"Saved feature column list to: {feature_columns_path}")


def main() -> None:
    """Run the complete dropout prediction workflow."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = load_dataset()
    print_original_dataset_note(df)
    validate_required_columns(df)
    print_initial_dataset_summary(df)

    df = remove_duplicate_rows(df)

    target_counts_before = df[TARGET_COLUMN].value_counts()
    print_target_counts(df, "Target Class Counts Before Filtering")
    plot_target_distribution_before_filtering(df)

    df_binary = filter_to_binary_classification(df)
    target_counts_after = df_binary[TARGET_COLUMN].value_counts()
    class_percentages = df_binary[TARGET_COLUMN].value_counts(normalize=True) * 100
    print_target_counts(df_binary, "Target Class Counts After Filtering")
    plot_target_distribution_after_filtering(df_binary)

    run_eda_plots(df_binary)
    pearson_r, pearson_p_value = print_pearson_correlation(df_binary)
    median_table = print_median_values_by_target(df_binary)

    df_model, target_encoder = encode_target_variable(df_binary)
    iqr_outlier_count = print_iqr_outlier_analysis(df_binary)

    X, y, feature_columns = separate_features_and_target(df_model)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_features(X, y)

    models, best_random_forest_params = train_models(X_train_scaled, y_train)
    results_df, predictions = evaluate_models(models, X_test_scaled, y_test)

    tuned_rf_predictions = predictions["Tuned Random Forest"]
    tuned_random_forest = models["Tuned Random Forest"]

    classification_report_text, _, _ = print_tuned_random_forest_report(
        y_test,
        tuned_rf_predictions,
        target_encoder,
    )
    plot_model_comparison_bar_chart(results_df)
    confusion_matrix_values = plot_confusion_matrix_tuned_random_forest(
        y_test,
        tuned_rf_predictions,
        target_encoder,
    )
    feature_importance_df = create_feature_importance(tuned_random_forest, feature_columns)

    save_streamlit_app_files(
        tuned_random_forest,
        scaler,
        target_encoder,
        feature_columns,
    )
    save_report_values(
        dataset_shape=df.shape,
        target_counts_before=target_counts_before,
        target_counts_after=target_counts_after,
        class_percentages=class_percentages,
        pearson_r=pearson_r,
        pearson_p_value=pearson_p_value,
        median_table=median_table,
        iqr_outlier_count=iqr_outlier_count,
        best_random_forest_params=best_random_forest_params,
        results_df=results_df,
        class_order=list(target_encoder.classes_),
        confusion_matrix_values=confusion_matrix_values,
        classification_report_text=classification_report_text,
        feature_importance_df=feature_importance_df,
    )

    print_section("Workflow Complete")
    print("Binary classification workflow completed: Dropout vs Graduate.")
    print("All plots, model results, and Streamlit app files have been saved.")


if __name__ == "__main__":
    main()

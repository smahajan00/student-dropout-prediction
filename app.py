"""
Streamlit app for the COMP 30044 CW2 student dropout prediction model.

The app loads the already-trained model artifacts and performs inference only.
It does not retrain the model.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_DIR / "student_dropout_model.pkl"
SCALER_PATH = PROJECT_DIR / "scaler.pkl"
ENCODER_PATH = PROJECT_DIR / "target_encoder.pkl"
FEATURE_COLUMNS_PATH = PROJECT_DIR / "feature_columns.pkl"

REQUIRED_FILES = {
    "student_dropout_model.pkl": MODEL_PATH,
    "scaler.pkl": SCALER_PATH,
    "target_encoder.pkl": ENCODER_PATH,
    "feature_columns.pkl": FEATURE_COLUMNS_PATH,
}


@st.cache_resource
def load_artifacts():
    """Load model artifacts from the current project directory."""
    missing_files = [
        filename for filename, path in REQUIRED_FILES.items() if not path.exists()
    ]

    if missing_files:
        missing_list = ", ".join(missing_files)
        raise FileNotFoundError(
            "Missing required model artifact file(s): "
            f"{missing_list}. Run student_dropout_prediction.py first."
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    target_encoder = joblib.load(ENCODER_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    return model, scaler, target_encoder, feature_columns


def set_feature_value(
    input_data: pd.DataFrame,
    feature_columns: list[str],
    column_name: str,
    value: float | int,
) -> None:
    """Fill a feature value only when the trained model contains that column."""
    if column_name in feature_columns:
        input_data.loc[0, column_name] = value


def build_input_dataframe(feature_columns: list[str], sidebar_values: dict[str, float | int]):
    """Create a one-row input DataFrame with all trained feature columns."""
    input_data = pd.DataFrame(0.0, index=[0], columns=feature_columns)

    set_feature_value(
        input_data,
        feature_columns,
        "Age at enrollment",
        sidebar_values["age_at_enrollment"],
    )
    set_feature_value(
        input_data,
        feature_columns,
        "Admission grade",
        sidebar_values["admission_grade"],
    )
    set_feature_value(
        input_data,
        feature_columns,
        "Previous qualification (grade)",
        sidebar_values["previous_qualification_grade"],
    )
    set_feature_value(
        input_data,
        feature_columns,
        "Curricular units 1st sem (approved)",
        sidebar_values["first_sem_approved"],
    )
    set_feature_value(
        input_data,
        feature_columns,
        "Curricular units 2nd sem (approved)",
        sidebar_values["second_sem_approved"],
    )
    set_feature_value(
        input_data,
        feature_columns,
        "Tuition fees up to date",
        sidebar_values["tuition_fees_up_to_date"],
    )
    set_feature_value(
        input_data,
        feature_columns,
        "Scholarship holder",
        sidebar_values["scholarship_holder"],
    )
    set_feature_value(
        input_data,
        feature_columns,
        "Debtor",
        sidebar_values["debtor"],
    )

    return input_data


def main() -> None:
    """Render the Streamlit prediction interface."""
    st.set_page_config(
        page_title="Student Dropout Prediction App",
        page_icon="🎓",
        layout="centered",
    )

    st.title("Student Dropout Prediction App")
    st.write(
        "This app predicts whether a student is likely to Dropout or Graduate "
        "based on selected student information."
    )
    st.warning(
        "This app is for educational demonstration only and should not be used "
        "as the only basis for real academic decisions."
    )

    try:
        model, scaler, target_encoder, feature_columns = load_artifacts()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()
    except Exception as error:
        st.error(f"Failed to load model artifacts: {error}")
        st.stop()

    st.sidebar.header("Student Information")

    sidebar_values = {
        "age_at_enrollment": st.sidebar.slider(
            "Age at enrollment",
            min_value=17,
            max_value=70,
            value=20,
        ),
        "admission_grade": st.sidebar.slider(
            "Admission Grade",
            min_value=0.0,
            max_value=200.0,
            value=120.0,
        ),
        "previous_qualification_grade": st.sidebar.slider(
            "Previous Qualification Grade",
            min_value=0.0,
            max_value=200.0,
            value=120.0,
        ),
        "first_sem_approved": st.sidebar.slider(
            "Curricular units 1st sem approved",
            min_value=0,
            max_value=30,
            value=5,
        ),
        "second_sem_approved": st.sidebar.slider(
            "Curricular units 2nd sem approved",
            min_value=0,
            max_value=30,
            value=5,
        ),
        "tuition_fees_up_to_date": st.sidebar.selectbox(
            "Tuition fees up to date",
            options=[1, 0],
        ),
        "scholarship_holder": st.sidebar.selectbox(
            "Scholarship holder",
            options=[0, 1],
        ),
        "debtor": st.sidebar.selectbox(
            "Debtor",
            options=[0, 1],
        ),
    }

    input_data = build_input_dataframe(feature_columns, sidebar_values)

    if st.button("Predict Student Outcome"):
        try:
            scaled_input = scaler.transform(input_data)
            scaled_input_df = pd.DataFrame(scaled_input, columns=feature_columns)

            prediction = model.predict(scaled_input_df)
            predicted_label = target_encoder.inverse_transform(prediction)[0]

            st.subheader("Predicted Outcome")
            st.write(f"**{predicted_label}**")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(scaled_input_df)[0]
                probability_table = pd.DataFrame(
                    {
                        "Outcome": target_encoder.classes_,
                        "Probability": probabilities,
                    }
                )
                st.subheader("Prediction Probabilities")
                st.dataframe(probability_table, hide_index=True)

            if predicted_label == "Dropout":
                st.error(
                    "The model predicts a higher risk of Dropout. Consider "
                    "academic advising, financial counselling, or mentoring "
                    "support for this student."
                )
            elif predicted_label == "Graduate":
                st.success("The model predicts that the student is likely to Graduate.")
            else:
                st.info(f"The model predicted: {predicted_label}")

        except Exception as error:
            st.error(f"Prediction failed: {error}")


if __name__ == "__main__":
    main()

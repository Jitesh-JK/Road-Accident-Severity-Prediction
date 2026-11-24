
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import base64


# 1. PAGE CONFIGURATION


st.set_page_config(
    page_title="Road Accident Severity Prediction",
    #page_icon="",
    layout="wide"
)


# 2. LOAD TRAINED MODEL


MODEL_PATH = (
    Path(__file__).parent
    / "artifacts"
    / "road_accident_model.joblib"
)

model_artifacts = joblib.load(MODEL_PATH)

model = model_artifacts["model"]
preprocessor = model_artifacts["preprocessor"]
feature_columns = model_artifacts["feature_columns"]


# 3. CATEGORY LABELS


SEVERITY_LABELS = {
    1: "Fatal",
    2: "Grievous Injury",
    3: "Minor Injury",
    4: "No Injury"
}

CATEGORY_OPTIONS = {
    "Day of Week": {
        1: "Sunday",
        2: "Monday",
        3: "Tuesday",
        4: "Wednesday",
        5: "Thursday",
        6: "Friday",
        7: "Saturday"
    },

    "Accident Location": {
        1: "Urban",
        2: "Rural"
    },

    "Road Feature": {
        1: "Single Lane",
        2: "Two Lanes",
        3: "Three or More Lanes Without Central Divider",
        4: "Four or More Lanes With Central Divider"
    },

    "Road Condition": {
        1: "Straight Road",
        2: "Slight Curve",
        3: "Sharp Curve",
        4: "Flat Road",
        5: "Gentle Incline",
        6: "Steep Incline",
        7: "Hump",
        8: "Dip"
    },

    "Weather": {
        1: "Fine",
        2: "Mist/Fog",
        3: "Cloud",
        4: "Light Rain",
        5: "Heavy Rain",
        6: "Hail/Sleet",
        7: "Snow",
        8: "Strong Wind",
        9: "Dust Storm",
        10: "Very Hot",
        11: "Very Cold",
        12: "Other Extraordinary Weather Conditions"
    },

    "Vehicle Type": {
        1: "Car/Jeep/Van",
        2: "SUV",
        3: "Bus",
        4: "Mini Bus",
        5: "Truck",
        6: "Two Wheeler",
        7: "Three Wheeler",
        8: "Cycle",
        9: "Pedestrian",
        10: "Tractor",
        11: "Unknown",
        12: "Animal",
        14: "LCV",
        15: "MAV"
    }
}

# --------------------------------------------------
# 4. PROJECT INTRODUCTION
# --------------------------------------------------

image_path = (
    Path(__file__).parent
    / "assets"
    / "accident.jpg"
)

with open(image_path, "rb") as image_file:
    encoded_image = base64.b64encode(
        image_file.read()
    ).decode()

st.markdown(
    f"""
    <style>
    .hero-section {{
        background-image:
            linear-gradient(
                rgba(0, 0, 0, 0.60),
                rgba(0, 0, 0, 0.60)
            ),
            url("data:image/jpeg;base64,{encoded_image}");

        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;

        padding: 85px 40px;
        border-radius: 15px;
        margin-bottom: 30px;
    }}

    .hero-section h1 {{
        color: white;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 15px;
    }}

    .hero-section p {{
        color: white;
        font-size: 18px;
    }}
    </style>

    <div class="hero-section">
        <h1>Road Accident Severity Prediction</h1>
        <p>
            An ML-based application that predicts accident
            severity using accident-related information.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

## 5. Model Performance Comparison 

st.header("Model Performance Comparison")

performance_data = {
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "Accuracy": [
        0.432,
        0.403,
        0.453
    ],

    "Macro F1": [
        0.310,
        0.344,
        0.349
    ],

    "Fatal Recall": [
        0.000,
        0.143,
        0.043
    ]
}

performance_df = pd.DataFrame(performance_data)

chart_data = performance_df.set_index("Model")

st.bar_chart(
    chart_data,
    y=["Accuracy", "Macro F1", "Fatal Recall"],
    height=400
)

st.caption(
    "The graph compares Accuracy, Macro F1 and Fatal Recall "
    "across the three model configurations."
)

st.divider()


# 6. ACCIDENT INFORMATION INPUT


st.header("Accident Information")

st.write("Enter the accident details below to predict its severity.")

with st.form("accident_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        day = st.selectbox(
            "Day of Week",
            options=list(CATEGORY_OPTIONS["Day of Week"].keys()),
            format_func=lambda x: CATEGORY_OPTIONS["Day of Week"][x]
        )

        location = st.selectbox(
            "Accident Location",
            options=list(CATEGORY_OPTIONS["Accident Location"].keys()),
            format_func=lambda x: CATEGORY_OPTIONS["Accident Location"][x]
        )

        chainage = st.number_input(
            "Accident Location Chainage (km)",
            min_value=0.0,
            value=125.5,
            step=0.1
        )

        road_feature = st.selectbox(
            "Road Feature",
            options=list(CATEGORY_OPTIONS["Road Feature"].keys()),
            format_func=lambda x: CATEGORY_OPTIONS["Road Feature"][x]
        )

        road_condition = st.selectbox(
            "Road Condition",
            options=list(CATEGORY_OPTIONS["Road Condition"].keys()),
            format_func=lambda x: CATEGORY_OPTIONS["Road Condition"][x]
        )

    with col2:
        weather = st.selectbox(
            "Weather Conditions",
            options=list(CATEGORY_OPTIONS["Weather"].keys()),
            format_func=lambda x: CATEGORY_OPTIONS["Weather"][x]
        )

        vehicle = st.selectbox(
            "Vehicle Type Involved",
            options=list(CATEGORY_OPTIONS["Vehicle Type"].keys()),
            format_func=lambda x: CATEGORY_OPTIONS["Vehicle Type"][x]
        )

        year = st.number_input(
            "Year",
            min_value=1900,
            max_value=2100,
            value=2023,
            step=1
        )

        month = st.selectbox(
            "Month",
            options=list(range(1, 13)),
            format_func=lambda x: pd.Timestamp(
                year=2000, month=x, day=1
            ).strftime("%B")
        )

        hour = st.number_input(
            "Hour (0–23)",
            min_value=0,
            max_value=23,
            value=12,
            step=1
        )

        minute = st.number_input(
            "Minute (0–59)",
            min_value=0,
            max_value=59,
            value=0,
            step=1
        )

    submitted = st.form_submit_button(
        "Predict Accident Severity",
        use_container_width=True
    )


# 7. PREDICTION RESULT

if "predicted_severity_code" not in st.session_state:
    st.session_state.predicted_severity_code = None

if submitted:

    input_data = pd.DataFrame([{
        "Day_of_Week": day,
        "Accident_Location_A": location,
        "Accident_Location_A_Chainage_km": chainage,
        "Road_Feature_E": road_feature,
        "Road_Condition_F": road_condition,
        "Weather_Conditions_H": weather,
        "Vehicle_Type_Involved_J_V1": vehicle,
        "Year": year,
        "Month": month,
        "Hour": hour,
        "Minute": minute
    }])

    input_data = input_data[feature_columns]

    try:
        processed_data = preprocessor.transform(input_data)
        prediction = int(model.predict(processed_data)[0])

        st.session_state.predicted_severity_code = prediction

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# Display four severity tiles
st.divider()
st.header("Accident Prediction Result")

st.write("Predicted severity category:")

severity_styles = {
    1: "Fatal",
    2: "Grievous Injury",
    3: "Minor Injury",
    4: "No Injury"
}

st.markdown(
    """
    <style>
    .severity-tile {
        background-color: white;
        color: #222222;
        border: 1px solid #dddddd;
        border-radius: 10px;
        padding: 24px 8px;
        text-align: center;
        min-height: 90px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 17px;
        font-weight: 600;
    }

    .severity-selected {
    background-color: #ffffff;
    color: #111111;
    border: 2px solid #222222;
    opacity: 1;
}

.severity-faded {
    background-color: #d1d1d1;
    color: #888888;
    border: 1px solid #bbbbbb;
    opacity: 0.20;
}
    </style>
    """,
    unsafe_allow_html=True
)

cols = st.columns(4)

for i, (code, label) in enumerate(severity_styles.items()):

    predicted = st.session_state.predicted_severity_code

    if predicted is None:
        tile_class = "severity-tile"
    elif predicted == code:
        tile_class = "severity-tile severity-selected"
    else:
        tile_class = "severity-tile severity-faded"

    with cols[i]:
        st.markdown(
            f"""
            <div class="{tile_class}">
                {label}
            </div>
            """,
            unsafe_allow_html=True
        )

if st.session_state.predicted_severity_code is not None:

    predicted_label = SEVERITY_LABELS[
        st.session_state.predicted_severity_code
    ]

    st.success(f"Predicted Severity: {predicted_label}")

    st.caption(
        "Prediction generated using the trained "
        "Balanced Random Forest model."
    )
else:
    st.info("Submit the accident details to highlight the predicted category.")


# # 8. DISCLAIMER


# st.divider()

# st.caption(
#     "Disclaimer: This application is an academic prototype. "
#     "Its predictions should not be used for real-world "
#     "or safety-critical decisions."
# )

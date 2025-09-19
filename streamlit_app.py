import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    """Load the trained model once and cache as a resource."""
    return joblib.load("trained_model.pkl")

model = load_model()
label_map = {0: "Extrovert", 1: "Introvert"}

st.title("Personality Prediction App")
st.markdown("Masukkan data kepribadian untuk memprediksi apakah seseorang ekstrovert atau introvert.")

# Input fields
time_spent_alone = st.number_input("Time Spent Alone (hours)", min_value=0.0, value=4.0, step=0.1)
stage_fear = st.number_input("Stage Fear", min_value=0, value=0, step=1)
social_event_attendance = st.number_input("Social Event Attendance", min_value=0.0, value=4.0, step=0.1)
going_outside = st.number_input("Going Outside", min_value=0.0, value=6.0, step=0.1)
drained_after_socializing = st.number_input(
    "Drained After Socializing (0 = No, 1 = Yes)",
    min_value=0, max_value=1, value=0, step=1
)
friends_circle_size = st.number_input("Friends Circle Size", min_value=0.0, value=13.0, step=1.0)
post_frequency = st.number_input("Post Frequency", min_value=0.0, value=5.0, step=1.0)

if st.button("Predict"):
    # Prepare input dataframe
    input_df = pd.DataFrame([{
        "Time_spent_Alone": time_spent_alone,
        "Stage_fear": stage_fear,
        "Social_event_attendance": social_event_attendance,
        "Going_outside": going_outside,
        "Drained_after_socializing": drained_after_socializing,
        "Friends_circle_size": friends_circle_size,
        "Post_frequency": post_frequency
    }])
    
    # Prediction
    pred = int(model.predict(input_df)[0])
    label = label_map.get(pred, "Unknown")
    
    # Display result
    st.success(f"Prediction: **{pred}** ({label})")

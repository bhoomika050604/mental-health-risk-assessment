import streamlit as st
import pandas as pd
import joblib
import requests
import tempfile

# ==========================
# Load model from Google Drive
# ==========================
def load_model_from_gdrive(file_id):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    if response.status_code != 200:
        st.error("❌ Failed to load model from Google Drive.")
        st.stop()
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(response.content)
        model = joblib.load(tmp_file.name)
    return model

PHQ_FILE_ID = "1ivMGK7g-ugz50xIitrte0Ka79YpLU_vz"
GAD_FILE_ID = "17hbrVSocX1CR0yJli8lSFADF-od5B1TL"

phq_model = load_model_from_gdrive(PHQ_FILE_ID)
gad_model = load_model_from_gdrive(GAD_FILE_ID)

# ==========================
# App Title
# ==========================
st.title("🧠 Mental Health Assessment App")
st.markdown("This app predicts your mental health risk using behavioral data and standard questionnaires.")

# ==========================
# Questionnaire options
# ==========================
options = {
    0: "Not at all",
    1: "Several days",
    2: "More than half the days",
    3: "Nearly every day"
}

def question_slider(label):
    return st.radio(label, options=[0, 1, 2, 3], format_func=lambda x: options[x], horizontal=True)

# ==========================
# PHQ-9 Section
# ==========================
st.header("📋Questionnaire (Depression Risk)")
phq_questions = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself — or that you are a failure",
    "Trouble concentrating on things",
    "Moving or speaking slowly, or being fidgety/restless",
    "Thoughts of self-harm or feeling better off dead"
]
phq_scores = [question_slider(f"PHQ{i+1}: {q}") for i, q in enumerate(phq_questions)]
phq_total = sum(phq_scores)

# ==========================
# GAD-7 Section
# ==========================
st.header("📋Questionnaire (Anxiety Risk)")
gad_questions = [
    "Feeling nervous, anxious or on edge",
    "Not being able to stop or control worrying",
    "Worrying too much about different things",
    "Trouble relaxing",
    "Being so restless that it is hard to sit still",
    "Becoming easily annoyed or irritable",
    "Feeling afraid as if something awful might happen"
]
gad_scores = [question_slider(f"GAD{i+1}: {q}") for i, q in enumerate(gad_questions)]
gad_total = sum(gad_scores)

# ==========================
# Behavioral Inputs
# ==========================
st.header("📊 Behavioral Data")
sleep_hours = st.slider("🛌 Average sleep hours", 0, 12, 7)
screen_time = st.slider("📱 Screen time per day (hours)", 0, 16, 6)
activity_level = st.slider("🏃 Activity level (1=low, 5=high)", 1, 5, 3)
social_interaction = st.slider("🗣️ Social interaction (1=low, 5=high)", 1, 5, 3)
stress_level = st.slider("😰 Stress level (1=low, 5=high)", 1, 5, 3)

# ==========================
# Risk Interpretation
# ==========================
def interpret_phq(score):
    if score < 5:
        return "🟢 Minimal"
    elif score < 10:
        return "🟡 Mild"
    elif score < 15:
        return "🟠 Moderate"
    elif score < 20:
        return "🟠 Moderately Severe"
    else:
        return "🔴 Severe"

def interpret_gad(score):
    if score < 5:
        return "🟢 Minimal"
    elif score < 10:
        return "🟡 Mild"
    elif score < 15:
        return "🟠 Moderate"
    else:
        return "🔴 Severe"

# ==========================
# Predict & Show Results
# ==========================
if st.button("🔍 Analyze Results"):
    input_data = pd.DataFrame({
        'sleep_hours': [sleep_hours],
        'screen_time': [screen_time],
        'activity_level': [activity_level],
        'social_interaction': [social_interaction],
        'stress_level': [stress_level]
    })

    pred_phq = phq_model.predict(input_data)[0]
    pred_gad = gad_model.predict(input_data)[0]

   
    st.subheader("📝 Predicted Risk")
    st.write(f"Depresion Risk: {interpret_phq(phq_total)}")
    st.write(f"Anxiety Risk: {interpret_gad(gad_total)}")

    st.subheader("📌 Recommendation")
    if phq_total >= 10 or gad_total >= 10:
        st.error("⚠️ Moderate or high mental health risk. Please consider professional support.")
    else:
        st.success("✅ Low risk. Continue maintaining your mental well-being!")


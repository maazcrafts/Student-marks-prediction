import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Marks Prediction",
    page_icon="🎓",
    layout="wide"
)

# ---------------- DATASET ----------------
data = {
    "Hours": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [20, 30, 40, 50, 60, 70, 80, 90, 100]
}

df = pd.DataFrame(data)

# ---------------- MODEL ----------------
X = df[["Hours"]]
y = df["Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

test_predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, test_predictions)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.block-container {
    max-width: 1150px;
    padding-top: 2rem;
}
.hero {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    padding: 32px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}
.hero h1 {
    font-size: 42px;
    margin: 0;
}
.hero p {
    font-size: 17px;
    margin-top: 8px;
}
.result {
    background: #eef2ff;
    border: 2px solid #6366f1;
    border-radius: 18px;
    padding: 28px;
    text-align: center;
    margin-top: 20px;
}
.marks {
    font-size: 48px;
    font-weight: 800;
}
.info-card {
    background: #ffffff;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #e5e7eb;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🎓 Student Marks")
    st.caption("Machine Learning Prediction System")
    st.divider()

    st.markdown("### Model")
    st.write("Linear Regression")

    st.markdown("### Dataset")
    st.write("9 sample records")

    st.markdown("### Input")
    st.write("Hours of Study")

    st.divider()
    st.success("Model Status: Ready")

# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <h1>🎓 Student Marks Prediction</h1>
    <p>Predict expected marks based on the number of hours a student studies.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- METRICS ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Dataset Records", "9")
c2.metric("Input Features", "1")
c3.metric("Model", "Linear Regression")
c4.metric("MAE", f"{mae:.2f}")

st.write("")

# ---------------- INPUT ----------------
st.markdown("### 📚 Enter Study Details")

hours = st.number_input(
    "Hours of Study",
    min_value=0.0,
    max_value=24.0,
    value=4.0,
    step=0.5
)

st.caption("Enter the approximate number of hours the student studies.")

if st.button("🎯 Predict Student Marks", use_container_width=True, type="primary"):
    prediction = model.predict([[hours]])[0]
    prediction = max(0, min(100, prediction))

    st.markdown(f"""
    <div class="result">
        <div>Predicted Marks</div>
        <div class="marks">{prediction:.1f} / 100</div>
        <div>Based on {hours:g} hour(s) of study</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- DATASET ----------------
st.write("")
with st.expander("📊 View Training Dataset"):
    st.dataframe(df, use_container_width=True)

# ---------------- MODEL INFO ----------------
st.write("")
st.markdown("### 📈 Model Information")

a, b = st.columns(2)

with a:
    st.markdown("""
    <div class="info-card">
    <b>Algorithm</b><br>
    Linear Regression<br><br>
    <b>Input Feature</b><br>
    Hours of Study<br><br>
    <b>Target</b><br>
    Marks
    </div>
    """, unsafe_allow_html=True)

with b:
    st.markdown("""
    <div class="info-card">
    <b>How it works</b><br>
    The model learns the relationship between study hours
    and marks from the training data and estimates the
    expected marks for the entered study time.
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.caption("Student Marks Prediction System • Python • Pandas • Scikit-learn • Linear Regression")

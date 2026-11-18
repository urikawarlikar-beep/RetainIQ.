
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(
    page_title="RetainIQ | Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- PROFESSIONAL THEME ----------
st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #e5e7eb;
}
[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #263449;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}
h1, h2, h3 {
    color: #f1f5f9 !important;
}
p, label, .stMarkdown {
    color: #cbd5e1;
}
[data-testid="stMetric"] {
    background: #151f31;
    border: 1px solid #29384e;
    padding: 20px;
    border-radius: 14px;
}
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
}
[data-testid="stMetricValue"] {
    color: #60a5fa !important;
}
.stButton > button, .stFormSubmitButton > button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 9px;
    padding: 0.6rem 1rem;
    font-weight: 600;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    background: #1d4ed8;
    color: white;
}
div[data-testid="stAlert"] {
    border-radius: 10px;
}
hr {
    border-color: #29384e;
}
</style>
""", unsafe_allow_html=True)


# ---------- LOAD PROJECT DATA ----------
@st.cache_resource
def load_model():
    return joblib.load("retainiq_model.joblib")


@st.cache_data
def load_data():
    data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    data.columns = data.columns.str.strip()
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"], errors="coerce"
    )
    data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})
    data = data.dropna(subset=["Churn"])
    return data


model = load_model()
df = load_data()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("# 📊 RetainIQ")
    st.caption("CUSTOMER INTELLIGENCE PLATFORM")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Executive Overview", "Customer Risk Lab", "Project Insights"]
    )

    st.divider()
    st.markdown("### About RetainIQ")
    st.caption(
        "Machine learning insights designed to help understand "
        "customer churn and explore retention opportunities."
    )
    st.caption("Models: Logistic Regression & Random Forest")
    st.caption("Dataset: Telco Customer Churn")


# ---------- HEADER ----------
st.markdown(
    "<h1 style='margin-bottom:0'>RetainIQ</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='font-size:17px;color:#94a3b8'>"
    "AI-Powered Customer Churn & Retention Intelligence</p>",
    unsafe_allow_html=True
)

st.divider()


# ---------- EXECUTIVE OVERVIEW ----------
if page == "Executive Overview":
    total_customers = len(df)
    churned = int(df["Churn"].sum())
    churn_rate = df["Churn"].mean() * 100
    avg_monthly = df["MonthlyCharges"].mean()
    avg_tenure = df["tenure"].mean()

    st.subheader("Executive Overview")
    st.caption("A high-level view of the customer portfolio.")

    a, b, c, d = st.columns(4)
    a.metric("Total Customers", f"{total_customers:,}")
    b.metric("Churned Customers", f"{churned:,}")
    c.metric("Historical Churn Rate", f"{churn_rate:.1f}%")
    d.metric("Avg. Monthly Charge", f"${avg_monthly:.2f}")

    st.write("")
    left, right = st.columns(2)

    with left:
        st.markdown("### Customer Churn Distribution")
        counts = df["Churn"].value_counts().reindex(
            [0, 1], fill_value=0
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("#151f31")
        ax.set_facecolor("#151f31")
        bars = ax.bar(
            ["Stayed", "Churned"],
            counts.values,
            color=["#38bdf8", "#fb7185"],
            width=0.55
        )
        ax.set_ylabel("Number of Customers", color="#cbd5e1")
        ax.tick_params(colors="#cbd5e1")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.bar_label(bars, color="white", padding=4)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown("### Churn Rate by Contract")
        contract_rates = (
            df.groupby("Contract", observed=False)["Churn"].mean() * 100
        ).sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("#151f31")
        ax.set_facecolor("#151f31")
        bars = ax.barh(
            contract_rates.index,
            contract_rates.values,
            color="#818cf8"
        )
        ax.set_xlabel("Historical Churn Rate (%)", color="#cbd5e1")
        ax.tick_params(colors="#cbd5e1")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.bar_label(bars, fmt="%.1f%%", color="white", padding=4)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("### Monthly Charges: Churn vs Retention")
    charge_summary = (
        df.groupby("Churn")["MonthlyCharges"].mean()
        .reindex([0, 1])
    )
    charge_summary.index = ["Stayed", "Churned"]
    st.bar_chart(charge_summary)

    st.info(
        "These are historical dataset statistics. They describe the "
        "provided data and should not be interpreted as live company metrics."
    )


# ---------- CUSTOMER RISK LAB ----------
elif page == "Customer Risk Lab":
    st.subheader("Customer Risk Lab")
    st.caption(
        "Enter a customer profile to estimate churn probability "
        "using the trained machine learning pipeline."
    )

    with st.form("risk_form"):
        st.markdown("#### 1. Customer Profile")
        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox(
                "Senior Citizen", [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
            partner = st.selectbox("Has Partner", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents", ["Yes", "No"])

        with c2:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes"] if phone == "Yes"
                else ["No phone service"]
            )
            internet = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"]
            )

        with c3:
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )
            monthly = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                max_value=200.0,
                value=70.0,
                step=1.0
            )

        st.markdown("#### 2. Services")
        s1, s2, s3 = st.columns(3)

        with s1:
            online_security = st.selectbox(
                "Online Security",
                ["No internet service"] if internet == "No"
                else ["No", "Yes"]
            )
            online_backup = st.selectbox(
                "Online Backup",
                ["No internet service"] if internet == "No"
                else ["No", "Yes"]
            )

        with s2:
            device_protection = st.selectbox(
                "Device Protection",
                ["No internet service"] if internet == "No"
                else ["No", "Yes"]
            )
            tech_support = st.selectbox(
                "Tech Support",
                ["No internet service"] if internet == "No"
                else ["No", "Yes"]
            )

        with s3:
            streaming_tv = st.selectbox(
                "Streaming TV",
                ["No internet service"] if internet == "No"
                else ["No", "Yes"]
            )
            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["No internet service"] if internet == "No"
                else ["No", "Yes"]
            )

        submitted = st.form_submit_button(
            "Estimate Churn Risk", use_container_width=True
        )

    if submitted:
        if phone == "No":
            multiple_lines = "No phone service"

        total_charges = monthly * tenure

        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total_charges
        }])

        probability = float(model.predict_proba(customer)[0][1])

        st.divider()
        st.markdown("### Risk Assessment Result")
        r1, r2 = st.columns([1, 2])

        with r1:
            st.metric("Estimated Churn Probability", f"{probability:.1%}")
            st.progress(probability)

        with r2:
            if probability >= 0.65:
                st.error("HIGH ESTIMATED RISK")
                st.write(
                    "Suggested next step: review service concerns and "
                    "consider an appropriate retention offer."
                )
            elif probability >= 0.35:
                st.warning("MODERATE ESTIMATED RISK")
                st.write(
                    "Suggested next step: review customer satisfaction, "
                    "billing experience, and service usage."
                )
            else:
                st.success("LOW ESTIMATED RISK")
                st.write(
                    "Suggested next step: maintain service quality and "
                    "continue regular customer engagement."
                )

        st.caption(
            "The risk bands are illustrative, not validated business "
            "thresholds. The estimate is not a guarantee of future behaviour."
        )


# ---------- PROJECT INSIGHTS ----------
else:
    st.subheader("Project Insights")
    st.caption("Methodology, model evaluation, and project limitations.")

    st.markdown("### Model Comparison")
    try:
        results = pd.read_csv("model_comparison.csv")
        display_results = results.copy()

        for column in [
            "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"
        ]:
            if column in display_results.columns:
                display_results[column] = display_results[column].map(
                    lambda value: f"{value:.2%}"
                )

        st.dataframe(display_results, use_container_width=True)
    except FileNotFoundError:
        st.warning("Run train_model.py to generate model_comparison.csv.")

    st.markdown("### Methodology")
    st.markdown("""
    1. **Data preparation:** Clean numeric charges and handle missing values.
    2. **Exploratory analysis:** Examine churn distribution and billing patterns.
    3. **Preprocessing:** Scale numeric features and encode categorical data.
    4. **Model training:** Train Logistic Regression and Random Forest.
    5. **Evaluation:** Compare accuracy, precision, recall, F1 score and ROC-AUC.
    6. **Prediction:** Use the saved preprocessing and model pipeline for
       customer risk estimation.
    """)

    st.markdown("### Limitations")
    st.markdown("""
    - Model performance depends on the quality and representativeness of the dataset.
    - The customer form uses an approximate total-charge calculation.
    - Risk categories are illustrative and require business validation.
    - Predictions should support, not replace, human judgement.
    """)

    st.markdown("### Data Source")
    st.markdown(
        "[Telco Customer Churn Dataset — Kaggle]"
        "(https://www.kaggle.com/datasets/blastchar/telco-customer-churn)"
    )

st.divider()
st.caption(
    "RetainIQ • Machine Learning Project • For educational and "
    "demonstration purposes only"
)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io

st.set_page_config(
    page_title="Invoice Payment Anomaly Detection",
    page_icon="📄",
    layout="wide"
)

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>

.main{
    background-color:#F8FAFC;
}

.metric-card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 10px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.title("📄 Invoice Payment Anomaly Detection")
st.caption("AI-Powered Detection of Suspicious Invoice Payments")

st.markdown("---")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Settings")

threshold = st.sidebar.slider(
    "Anomaly Threshold",
    0.50,
    1.00,
    0.80,
    0.01
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Invoice CSV",
    type=["csv"]
)

# -----------------------------
# Generate Demo Dataset
# -----------------------------
def generate_demo():

    np.random.seed(42)

    n = 100

    df = pd.DataFrame({
        "Invoice ID":[f"INV-{1000+i}" for i in range(n)],
        "Vendor":np.random.choice(
            [
                "ABC Ltd",
                "Tech Corp",
                "Global Traders",
                "Blue Systems",
                "Prime Supplies"
            ],
            n
        ),
        "Amount":np.random.randint(5000,100000,n),
        "Payment Days":np.random.randint(1,60,n)
    })

    scores=np.random.uniform(0.05,0.98,n)

    df["Anomaly Score"]=scores

    df["Status"]=np.where(
        df["Anomaly Score"]>=threshold,
        "Anomaly",
        "Normal"
    )

    return df

# -----------------------------
# Data
# -----------------------------
if uploaded_file:

    df=pd.read_csv(uploaded_file)

    if "Anomaly Score" not in df.columns:

        np.random.seed(1)

        df["Anomaly Score"]=np.random.uniform(
            0.05,
            0.98,
            len(df)
        )

    df["Status"]=np.where(
        df["Anomaly Score"]>=threshold,
        "Anomaly",
        "Normal"
    )

else:

    df=generate_demo()

# -----------------------------
# Metrics
# -----------------------------
total=len(df)

anomaly=df[df["Status"]=="Anomaly"]

normal=df[df["Status"]=="Normal"]

c1,c2,c3,c4=st.columns(4)

c1.metric("Invoices",total)

c2.metric("Normal",len(normal))

c3.metric("Anomalies",len(anomaly))

if total>0:
    c4.metric(
        "Anomaly %",
        f"{len(anomaly)/total*100:.1f}%"
    )

st.markdown("---")

# -----------------------------
# Charts
# -----------------------------
left,right=st.columns([2,1])

with left:

    st.subheader("Invoice Amount Distribution")

    fig=px.histogram(
        df,
        x="Amount",
        color="Status",
        nbins=30
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with right:

    st.subheader("Detection Summary")

    fig2=px.pie(
        df,
        names="Status",
        hole=0.6
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# -----------------------------
# Scatter
# -----------------------------
st.subheader("Payment Behaviour")

fig3=px.scatter(
    df,
    x="Payment Days",
    y="Amount",
    color="Status",
    size="Anomaly Score",
    hover_data=["Vendor","Invoice ID"]
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# -----------------------------
# High Risk Table
# -----------------------------
st.subheader("🚨 High Risk Invoices")

risk=anomaly.sort_values(
    "Anomaly Score",
    ascending=False
)

st.dataframe(
    risk,
    use_container_width=True
)

# -----------------------------
# Full Dataset
# -----------------------------
with st.expander("View Complete Dataset"):

    st.dataframe(
        df,
        use_container_width=True
    )

# -----------------------------
# Download
# -----------------------------
csv=df.to_csv(index=False).encode()

st.download_button(
    "⬇ Download Report",
    csv,
    "invoice_anomaly_report.csv",
    "text/csv"
)

st.markdown("---")

st.success("✅ Demo application only. Results are simulated for UI demonstration purposes.")
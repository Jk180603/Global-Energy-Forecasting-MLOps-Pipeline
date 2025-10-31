import streamlit as st
import requests

st.title("EnergyGlobal – AI Energy Forecaster")

pop = st.slider("Population (M)", 1, 1500, 80)
gdp = st.slider("GDP (T$)", 0.1, 30.0, 4.0)
gpc = st.slider("GDP/capita ($)", 500, 100000, 50000)
gr = st.slider("Growth Rate (%)", -5.0, 15.0, 2.5)

if st.button("Forecast"):
    payload = {"population": pop, "gdp": gdp, "gdp_per_capita": gpc, "growth_rate": gr}
    try:
        r = requests.post("http://localhost:8000/forecast", json=payload).json()
        st.success(f"**Predicted:** {r['predicted_TWh']} TWh")
        st.metric("Savings", f"{r['savings_TWh']} TWh")
        st.info(r['advice'])
    except:
        st.error("API not running")

st.sidebar.title("Advanced Mode")
mode = st.sidebar.radio("Choose", ["Simple Forecast", "Germany LSTM Forecast"])

if mode == "Germany LSTM Forecast":
    st.header("Germany 5-Year Forecast (LSTM)")
    if st.button("Run LSTM Forecast"):
        # Mock forecast
        years = [2025, 2026, 2027, 2028, 2029]
        values = [580, 595, 610, 620, 635]
        chart_data = pd.DataFrame({"Year": years, "TWh": values})
        st.line_chart(chart_data)
        st.success("Germany will consume ~635 TWh by 2029")

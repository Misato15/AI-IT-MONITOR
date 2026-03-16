import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI IT Monitor", layout="wide")

st.title("AI Infrastructure Monitoring Dashboard")
st.write("Machine Learning prototype for predictive IT incident detection")

# -------------------------
# DATASET DE ENTRENAMIENTO
# -------------------------

data = {
    "cpu":[90,85,70,30,20,95,60,40,88,92,55,45],
    "ram":[88,80,75,40,30,90,65,35,82,87,50,42],
    "network":[80,75,60,20,10,85,55,30,78,81,40,33],
    "processes":[220,210,180,90,70,240,150,110,200,215,120,100],
    "incident":[1,1,1,0,0,1,0,0,1,1,0,0]
}

df = pd.DataFrame(data)

X = df[["cpu","ram","network","processes"]]
y = df["incident"]

model = RandomForestClassifier(random_state=42)
model.fit(X,y)

# -------------------------
# SIDEBAR CONTROLES
# -------------------------

st.sidebar.header("System Metrics")

cpu = st.sidebar.slider("CPU Usage %",0,100,60)
ram = st.sidebar.slider("RAM Usage %",0,100,55)
network = st.sidebar.slider("Network Load %",0,100,50)
processes = st.sidebar.slider("Active Processes",50,300,150)

input_data = pd.DataFrame({
    "cpu":[cpu],
    "ram":[ram],
    "network":[network],
    "processes":[processes]
})

# -------------------------
# PREDICCIÓN IA
# -------------------------

prediction = model.predict(input_data)
probability = model.predict_proba(input_data)

# -------------------------
# MÉTRICAS
# -------------------------

col1,col2,col3,col4 = st.columns(4)

col1.metric("CPU Usage",f"{cpu}%")
col2.metric("RAM Usage",f"{ram}%")
col3.metric("Network Load",f"{network}%")
col4.metric("Processes",processes)

st.divider()

# -------------------------
# RESULTADO IA
# -------------------------

st.subheader("Incident Risk Analysis")

risk = probability[0][1] * 100

st.progress(int(risk))
st.write(f"Risk Score: {risk:.1f}%")

if prediction[0] == 1:
    st.error(f"⚠ High Incident Risk ({risk:.1f}%)")
else:
    st.success("System Stable")

# -------------------------
# HISTORIAL SIMULADO
# -------------------------

st.subheader("System Metrics History")

time = np.arange(0,20)

cpu_history = np.clip(np.random.normal(cpu,5,20),0,100)
ram_history = np.clip(np.random.normal(ram,5,20),0,100)
network_history = np.clip(np.random.normal(network,5,20),0,100)

history_df = pd.DataFrame({
    "time":time,
    "CPU":cpu_history,
    "RAM":ram_history,
    "Network":network_history
})

st.line_chart(history_df.set_index("time"))

# -------------------------
# GRAFICA DE CARGA
# -------------------------

st.subheader("Current System Load")

labels = ["CPU","RAM","Network"]
values = [cpu,ram,network]

fig, ax = plt.subplots()

ax.bar(labels,values)

ax.set_ylabel("Usage %")
ax.set_title("Current Infrastructure Load")

st.pyplot(fig)

# -------------------------
# EVENT LOG
# -------------------------

st.subheader("🚨 Incident Event Log")

events = [
    "High CPU spike detected",
    "Network anomaly detected",
    "Unauthorized access attempt",
    "Server overheating",
    "Memory usage critical",
    "Disk I/O spike detected",
    "Service response delay detected"
]

num_events = 5

if prediction[0] == 1:
    num_events = 8

event_log = []

for i in range(num_events):
    event = random.choice(events)
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    event_log.append({
        "Time":timestamp,
        "Event":event
    })

log_df = pd.DataFrame(event_log)

st.table(log_df)

# -------------------------
# DATASET
# -------------------------

with st.expander("Training Dataset Used by ML Model"):
    st.dataframe(df)
    
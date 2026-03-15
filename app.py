import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI IT Monitor", layout="wide")

st.title("AI Infrastructure Monitoring Dashboard")

st.write("Machine Learning prototype for predictive IT incident detection")

# -------------------------
# Dataset de entrenamiento
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

model = RandomForestClassifier()
model.fit(X,y)

# -------------------------
# Sidebar controles
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
# Predicción IA
# -------------------------

prediction = model.predict(input_data)
probability = model.predict_proba(input_data)

# -------------------------
# Métricas principales
# -------------------------

col1,col2,col3,col4 = st.columns(4)

col1.metric("CPU Usage",f"{cpu}%")
col2.metric("RAM Usage",f"{ram}%")
col3.metric("Network Load",f"{network}%")
col4.metric("Processes",processes)

st.divider()

# -------------------------
# Resultado de IA
# -------------------------

st.subheader("Incident Risk Analysis")

if prediction[0] == 1:
    st.error(f"⚠ High Incident Risk ({probability[0][1]*100:.1f}%)")
else:
    st.success(f"System Stable ({probability[0][0]*100:.1f}%)")

# -------------------------
# Historial simulado
# -------------------------

st.subheader("System Metrics History")

time = np.arange(0,20)

cpu_history = np.random.normal(cpu,5,20)
ram_history = np.random.normal(ram,5,20)
network_history = np.random.normal(network,5,20)

history_df = pd.DataFrame({
    "time":time,
    "CPU":cpu_history,
    "RAM":ram_history,
    "Network":network_history
})

st.line_chart(history_df.set_index("time"))

# -------------------------
# Grafica actual
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
# Dataset entrenamiento
# -------------------------

with st.expander("Training Dataset Used by ML Model"):
    st.dataframe(df)

import random
from datetime import datetime

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

# generar eventos simulados
event_log = []

for i in range(5):
    event = random.choice(events)
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    event_log.append({
        "Time":timestamp,
        "Event":event
    })

log_df = pd.DataFrame(event_log)

st.table(log_df)
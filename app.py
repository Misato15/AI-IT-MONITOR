import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI IT Monitor", layout="wide")

st.title("Dashboard de monitoreo de infraestructura con IA")
st.write(" Prototipo de Machine Learning para detección predictiva de incidentes IT")

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

st.sidebar.header("Metricas del Sistema")

cpu = st.sidebar.slider("Uso de CPU  %",0,100,60)
ram = st.sidebar.slider("Uso de RAM %",0,100,55)
network = st.sidebar.slider("Carga de Red %",0,100,50)
processes = st.sidebar.slider("Procesos Activos",50,300,150)

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

col1.metric("Uso de CPU",f"{cpu}%")
col2.metric("Uso de RAM",f"{ram}%")
col3.metric("Carga de Red",f"{network}%")
col4.metric("Procesos Activos",processes)

st.divider()

# -------------------------
# RESULTADO IA
# -------------------------

st.subheader("Analisis de incidentes de riesgo")

risk = probability[0][1] * 100

st.progress(int(risk))
st.write(f"Score de Riesgo: {risk:.1f}%")

if prediction[0] == 1:
    st.error(f"⚠ Incidente de alto riesgo ({risk:.1f}%)")
else:
    st.success("Sistema Estable (Bajo riesgo)")

# -------------------------
# HISTORIAL SIMULADO
# -------------------------

st.subheader("Historial de Métricas del Sistema")

time = np.arange(0,20)

cpu_history = np.clip(np.random.normal(cpu,5,20),0,100)
ram_history = np.clip(np.random.normal(ram,5,20),0,100)
network_history = np.clip(np.random.normal(network,5,20),0,100)

history_df = pd.DataFrame({
    "Tiempo":time,
    "CPU":cpu_history,
    "RAM":ram_history,
    "Red":network_history
})

st.line_chart(history_df.set_index("Tiempo"))

# -------------------------
# GRAFICA DE CARGA
# -------------------------

st.subheader("Carga Actual del Sistema")

labels = ["CPU","RAM","Red"]
values = [cpu,ram,network]

fig, ax = plt.subplots()

ax.bar(labels,values)

ax.set_ylabel("Uso %")
ax.set_title("Carga de Infraestructura Actual")

st.pyplot(fig)

# -------------------------
# EVENT LOG
# -------------------------

st.subheader(" Registro de Eventos de Incidente")

eventos = [
    "Pico alto de CPU detectado",
    "Anomalía en la red detectada",
    "Intento de acceso no autorizado",
    "Sobrecalentamiento del servidor",
    "Uso de memoria crítico",
    "Pico de E/S de disco detectado",
    "Retraso en la respuesta del servicio detectado"
]

num_events = 5

if prediction[0] == 1:
    num_events = 8

event_log = []

for i in range(num_events):
    event = random.choice(events)
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    event_log.append({
        "Tiempo":timestamp,
        "Evento":event
    })

log_df = pd.DataFrame(event_log)

st.table(log_df)

# -------------------------
# DATASET
# -------------------------

with st.expander("Dataset de Entrenamiento Utilizado por el Modelo ML"):
    st.dataframe(df)

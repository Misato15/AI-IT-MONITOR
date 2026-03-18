import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime
import psutil
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI IT Monitor PRO", layout="wide")

st.title("🚀 Sistema Inteligente de Monitoreo de Infraestructura TI")

# -------------------------
# DATASET
# -------------------------

data = {
    "cpu":[90,85,70,30,20,95,60,40,88,92,55,45,78,82,67,73],
    "ram":[88,80,75,40,30,90,65,35,82,87,50,42,70,76,60,68],
    "network":[80,75,60,20,10,85,55,30,78,81,40,33,65,70,50,58],
    "processes":[220,210,180,90,70,240,150,110,200,215,120,100,170,185,140,160],
    "incident":[1,1,1,0,0,1,0,0,1,1,0,0,1,1,0,0]
}

df = pd.DataFrame(data)

X = df[["cpu","ram","network","processes"]]
y = df["incident"]

rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier()

rf.fit(X,y)
gb.fit(X,y)

# -------------------------
# SIDEBAR
# -------------------------

st.sidebar.header("⚙️ Configuración")

modo = st.sidebar.selectbox("Modo de datos", ["Simulado","Tiempo Real"])
modelo = st.sidebar.selectbox("Modelo IA", ["Random Forest","Gradient Boosting"])

servers = ["Server A","Server B","Server C"]
server = st.sidebar.selectbox("Servidor", servers)

# BOTÓN ACTUALIZAR
if st.sidebar.button("🔄 Actualizar datos"):
    st.sidebar.success("Datos actualizados")
    st.rerun()

# -------------------------
# INPUTS
# -------------------------

if modo == "Simulado":
    cpu = st.sidebar.slider("CPU %",0,100,60)
    ram = st.sidebar.slider("RAM %",0,100,55)
    network = st.sidebar.slider("Red %",0,100,50)
    processes = st.sidebar.slider("Procesos",50,300,150)

else:
    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory().percent
    processes = len(psutil.pids())
    network = np.random.randint(20,80)
    st.sidebar.success("📡 Datos en tiempo real")

# variación leve por servidor
cpu += random.randint(-5,5)
ram += random.randint(-5,5)

input_data = pd.DataFrame({
    "cpu":[cpu],
    "ram":[ram],
    "network":[network],
    "processes":[processes]
})

# -------------------------
# MODELO
# -------------------------

model = rf if modelo == "Random Forest" else gb

prediction = model.predict(input_data)
prob = model.predict_proba(input_data)

risk = prob[0][1]*100
future_risk = risk + np.random.randint(-5,10)

# -------------------------
# MÉTRICAS
# -------------------------

col1,col2,col3,col4 = st.columns(4)

col1.metric("CPU",f"{cpu:.1f}%")
col2.metric("RAM",f"{ram:.1f}%")
col3.metric("Red",f"{network:.1f}%")
col4.metric("Procesos",processes)

st.divider()

# -------------------------
# RIESGO
# -------------------------

st.subheader("🧠 Análisis Inteligente")

st.progress(int(risk))
st.write(f"Riesgo actual: {risk:.1f}%")
st.write(f"Predicción futura: {future_risk:.1f}%")

# SEMÁFORO

if risk > 70:
    st.markdown("## 🔴 CRÍTICO")
    st.error("Fallo inminente detectado")
elif risk > 50:
    st.markdown("## 🟠 MEDIO")
    st.warning("Riesgo elevado")
else:
    st.markdown("## 🟢 ESTABLE")
    st.success("Sistema estable")

# -------------------------
# RECOMENDACIONES
# -------------------------

st.subheader("🛠 Recomendaciones")

if cpu > 85:
    st.write("• Reducir carga de CPU")
if ram > 80:
    st.write("• Optimizar memoria RAM")
if network > 75:
    st.write("• Revisar tráfico de red")
if risk < 50:
    st.write("• Sistema funcionando correctamente")

# -------------------------
# HISTORIAL
# -------------------------

st.subheader("📊 Historial")

time = np.arange(20)

cpu_hist = np.clip(np.random.normal(cpu,5,20),0,100)
ram_hist = np.clip(np.random.normal(ram,5,20),0,100)
net_hist = np.clip(np.random.normal(network,5,20),0,100)

hist = pd.DataFrame({
    "Tiempo":time,
    "CPU":cpu_hist,
    "RAM":ram_hist,
    "Red":net_hist
})

st.line_chart(hist.set_index("Tiempo"))

# -------------------------
# EVENTOS
# -------------------------

st.subheader("📋 Eventos del Sistema")

events = [
    "CPU alta",
    "Anomalía de red",
    "Acceso sospechoso",
    "Sobrecalentamiento",
    "Memoria crítica"
]

num = 5 if risk < 50 else 8

log = []

for i in range(num):
    log.append({
        "Hora": datetime.now().strftime("%H:%M:%S"),
        "Evento": random.choice(events)
    })

st.dataframe(pd.DataFrame(log), use_container_width=True)

# -------------------------
# INFO MODELO
# -------------------------

with st.expander("ℹ️ Información del modelo"):
    st.write("Modelo seleccionado:", modelo)
    st.write("Variables utilizadas: CPU, RAM, Red, Procesos")

# -------------------------
# DATASET
# -------------------------

with st.expander("📂 Dataset de entrenamiento"):
    st.dataframe(df)
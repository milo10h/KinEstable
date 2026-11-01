import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

R = 8.314  # J/mol·K

st.set_page_config(page_title="KinEstable", page_icon="💊", layout="wide")
st.title("💊 KinEstable")
st.subheader("Herramienta de Análisis Cinético y Predicción de Estabilidad de Productos Farmacéuticos")

c1, c2, c3 = st.columns(3)
farmaco = c1.text_input("Principio activo", "Rosuvastatina cálcica")
unidad = c2.selectbox("Unidad de tiempo", ["horas", "días", "semanas", "meses"], index=1)
unidad_conc = c3.selectbox("Unidad de concentración", ["% remanente", "mg/mL", "µg/mL", "mol/L"])

def tiempos(orden, k, A0):
    """Tiempo hasta que queda el x% de la concentración inicial."""
    res = {}
    for x in [50, 90, 95, 98]:
        f = x / 100
        if orden == "Orden 0":
            res[f"t{x}"] = A0 * (1 - f) / k
        elif orden == "Orden 1":
            res[f"t{x}"] = np.log(1 / f) / k
        else:
            res[f"t{x}"] = (1 / f - 1) / (k * A0)
    return res

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Datos", "📈 Gráficos", "⚗️ Orden y parámetros",
    "🌡️ Módulo Arrhenius", "🔮 Predicción", "🤖 Conclusiones"
])

# ---------- DATOS ----------
with tab1:
    st.header(f"Datos experimentales de {farmaco}")
    st.write("Ingresa o edita tus datos. Puedes agregar filas abajo. "
             "Los valores que vienen son solo de ejemplo.")
    datos_ejemplo = pd.DataFrame({
        "Tiempo": [0, 10, 20, 30, 40, 50],
        "Concentracion": [100, 90.5, 81.9, 74.1, 67.0, 60.7]
    })
    datos = st.data_editor(datos_ejemplo, num_rows="dynamic", key="datos")

datos = datos.dropna()
t = datos["Tiempo"].astype(float).values
A = datos["Concentracion"].astype(float).values

modelos = {
    "Orden 0": (A, "[A]"),
    "Orden 1": (np.log(A), "ln[A]"),
    "Orden 2": (1 / A, "1/[A]")
}
resultados = {}
for nombre, (y, etiqueta) in modelos.items():
    reg = stats.linregress(t, y)
    resultados[nombre] = {"pendiente": reg.slope, "intercepto": reg.intercept,
                          "R2": reg.rvalue ** 2, "y": y, "etiqueta": etiqueta}

orden = max(resultados, key=lambda m: resultados[m]["R2"])
p = resultados[orden]
if orden == "Orden 0":
    k = -p["pendiente"]; A0 = p["intercepto"]
elif orden == "Orden 1":
    k = -p["pendiente"]; A0 = np.exp(p["intercepto"])
else:
    k = p["pendiente"]; A0 = 1 / p["intercepto"]
tiempos_exp = tiempos(orden, k, A0)

# ---------- GRÁFICOS ----------
with tab2:
    st.header("Gráficos por el método integral")
    grafico = st.radio("¿Qué gráfico quieres ver?",
                       ["Orden 0", "Orden 1", "Orden 2", "Todos"], horizontal=True)
    lista = list(resultados) if grafico == "Todos" else [grafico]
    cols = st.columns(len(lista))
    for i, nombre in enumerate(lista):
        r = resultados[nombre]
        fig, ax = plt.subplots()
        ax.scatter(t, r["y"])
        ax.plot(t, r["intercepto"] + r["pendiente"] * t, "r")
        ax.set_xlabel(f"t ({unidad})")
        ax.set_ylabel(r["etiqueta"])
        ax.set_title(f"{farmaco}: {r['etiqueta']} vs t")
        with cols[i]:
            st.pyplot(fig)
            st.metric("R²", f"{r['R2']:.5f}")
            st.write(f"Pendiente: {r['pendiente']:.5f}")
            st.write(f"Intercepto: {r['intercepto']:.5f}")

# ---------- ORDEN Y PARÁMETROS ----------
with tab3:
    st.header("Orden de reacción")
    st.dataframe(pd.DataFrame({m: [resultados[m]["R2"]] for m in resultados},
                              index=["R²"]).round(5))
    st.success(f"El modelo más probable para {farmaco} es **{orden}** (R² = {p['R2']:.5f})")

    st.header("Parámetros cinéticos")
    opcion = st.selectbox("¿Qué quieres calcular?",
                          ["Todos", "Constante k", "t50", "t90", "t95", "t98"])
    if st.button("Calcular", key="btn_param"):
        if opcion in ["Todos", "Constante k"]:
            st.metric("Constante cinética k", f"{k:.5f}")
        if opcion == "Todos":
            c = st.columns(4)
            for i, (nom, val) in enumerate(tiempos_exp.items()):
                c[i].metric(nom, f"{val:.2f} {unidad}")
        elif opcion != "Constante k":
            st.metric(opcion, f"{tiempos_exp[opcion]:.2f} {unidad}")
        st.caption("t50, t90, t95 y t98 = tiempo hasta que queda el 50%, 90%, 95% y 98% "
                   "del principio activo. El t90 corresponde a la vida útil.")

# ---------- ARRHENIUS ----------
with tab4:
    st.header("Módulo Arrhenius")
    st.write(f"Ingresa las constantes k de {farmaco} obtenidas a diferentes temperaturas (°C).")
    arr_ejemplo = pd.DataFrame({
        "Temperatura_C": [40, 50, 60, 70],
        "k": [0.005, 0.012, 0.027, 0.058]
    })
    arr = st.data_editor(arr_ejemplo, num_rows="dynamic", key="arr").dropna()

T = arr["Temperatura_C"].astype(float).values + 273.15
kT = arr["k"].astype(float).values
x_arr = 1 / T
y_arr = np.log(kT)
reg_a = stats.linregress(x_arr, y_arr)
Ea = -reg_a.slope * R
A_pre = np.exp(reg_a.intercept)

def k_a(temp_c):
    return A_pre * np.exp(-Ea / (R * (temp_c + 273.15)))

with tab4:
    ver = st.selectbox("¿Qué quieres ver?",
                       ["Todo", "Tabla 1/T y ln(k)", "Gráfico de Arrhenius",
                        "Energía de activación", "Factor preexponencial"])
    if st.button("Mostrar", key="btn_arr"):
        if ver in ["Todo", "Tabla 1/T y ln(k)"]:
            st.dataframe(pd.DataFrame({"T (K)": T, "1/T": x_arr, "ln(k)": y_arr}))
        if ver in ["Todo", "Gráfico de Arrhenius"]:
            fig, ax = plt.subplots()
            ax.scatter(x_arr, y_arr)
            ax.plot(x_arr, reg_a.intercept + reg_a.slope * x_arr, "r")
            ax.set_xlabel("1/T (1/K)")
            ax.set_ylabel("ln(k)")
            ax.set_title(f"Gráfico de Arrhenius: {farmaco}")
            st.pyplot(fig)
            st.write(f"R² = {reg_a.rvalue**2:.5f}")
        if ver in ["Todo", "Energía de activación"]:
            st.metric("Energía de activación (Ea)", f"{Ea/1000:.2f} kJ/mol")
        if ver in ["Todo", "Factor preexponencial"]:
            st.metric("Factor preexponencial (A)", f"{A_pre:.3e}")

# ---------- PREDICCIÓN ----------
with tab5:
    st.header("Predicción de estabilidad")
    T_pred = st.slider("Elige la temperatura (°C)", -10.0, 100.0, 25.0, 0.5)
    k_pred = k_a(T_pred)
    tp = tiempos(orden, k_pred, A0)
    st.metric(f"k predicha a {T_pred} °C", f"{k_pred:.6f}")
    c = st.columns(4)
    for i, (nom, val) in enumerate(tp.items()):
        c[i].metric(nom, f"{val:.2f} {unidad}")

    st.subheader("Condiciones de almacenamiento farmacéutico")
    condiciones = {"Refrigeración (5 °C)": 5, "Ambiente controlado (25 °C)": 25,
                   "Zona climática IV (30 °C)": 30, "Acelerada (40 °C)": 40}
    tabla = pd.DataFrame([{"Condición": nom, "k": k_a(tc),
                           **{key: f"{v:.2f} {unidad}" for key, v in tiempos(orden, k_a(tc), A0).items()}}
                          for nom, tc in condiciones.items()])
    st.dataframe(tabla, hide_index=True)

# ---------- CONCLUSIONES ----------
with tab6:
    st.header("Conclusiones automáticas")
    if st.button("Generar conclusiones", key="btn_conc"):
        sensibilidad = "muy sensible" if Ea/1000 > 80 else "moderadamente sensible"
        t90_5 = tiempos(orden, k_a(5), A0)["t90"]
        t90_25 = tiempos(orden, k_a(25), A0)["t90"]
        t90_30 = tiempos(orden, k_a(30), A0)["t90"]
        factor = t90_25 / t90_30
        st.info(
            f"La degradación de {farmaco} se ajusta mejor a una cinética de {orden.lower()} "
            f"(R² = {p['R2']:.4f}), con una constante k = {k:.5f}. "
            f"La energía de activación es de {Ea/1000:.1f} kJ/mol, por lo que el principio activo "
            f"es {sensibilidad} a la temperatura.\n\n"
            f"Vida útil estimada (t90): {t90_5:.1f} {unidad} a 5 °C, {t90_25:.1f} {unidad} a 25 °C "
            f"y {t90_30:.1f} {unidad} a 30 °C. Almacenar a 25 °C en lugar de 30 °C prolonga la "
            f"vida útil {factor:.1f} veces.\n\n"
            f"A {T_pred} °C, {farmaco} conservaría el 90% de su contenido durante "
            f"{tp['t90']:.1f} {unidad} y se degradaría a la mitad en {tp['t50']:.1f} {unidad}."
        )
        st.caption("Estas predicciones asumen que el mecanismo de degradación no cambia con la "
                   "temperatura. Deben confirmarse con estudios de estabilidad reales.")
st.divider()
st.markdown(
    """
    
        KinEstable © 2026 · Todos los derechos reservados
        Desarrollado por: María Paula Herrera · Juan Camilo Hoyos · Carolina Rodríguez
        Universidad de Antioquia
    
    """,
    unsafe_allow_html=True
)

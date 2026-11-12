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
    col_titulo, col_boton = st.columns([3, 1])
    with col_titulo:
        st.header(f"Datos experimentales de {farmaco}")
    with col_boton:
        archivo = st.file_uploader("📂 Subir datos desde Excel", type=["xlsx", "csv"])
    st.write("Ingresa o edita tus datos en la tabla, o súbelos desde Excel. "
             "El archivo debe tener dos columnas: Tiempo y Concentracion.")
    datos_ejemplo = pd.DataFrame({
        "Tiempo": [0, 10, 20, 30, 40, 50],
        "Concentracion": [100, 90.5, 81.9, 74.1, 67.0, 60.7]
    })
    clave = "datos"
    if archivo is not None:
        try:
            if archivo.name.endswith(".csv"):
                subido = pd.read_csv(archivo)
            else:
                subido = pd.read_excel(archivo)
            subido = subido.iloc[:, :2]
            subido.columns = ["Tiempo", "Concentracion"]
            subido = subido.apply(pd.to_numeric, errors="coerce").dropna()
            datos_ejemplo = subido
            clave = "datos_" + archivo.name
            st.success(f"Datos cargados desde {archivo.name}")
        except Exception as e:
            st.error(f"No se pudo leer el archivo: {e}")
    datos = st.data_editor(datos_ejemplo, num_rows="dynamic", key=clave)

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
            signo = "+" if r["intercepto"] >= 0 else "−"
            st.write(f"**Ecuación:** {r['etiqueta']} = {r['pendiente']:.6g}·t {signo} {abs(r['intercepto']):.6g}")

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
    st.header("🤖 Conclusiones automáticas")
    if st.button("Generar conclusiones", key="btn_conc"):
        a_dias = {"horas": 1/24, "días": 1, "semanas": 7, "meses": 30.4}[unidad]
        def meses(x):
            return x * a_dias / 30.4

        # 1. Calidad de los datos
        r2_orden = sorted([resultados[m]["R2"] for m in resultados], reverse=True)
        diferencia = r2_orden[0] - r2_orden[1]
        degradado = (A[0] - A.min()) / A[0] * 100
        if p["R2"] >= 0.99:
            calidad = "excelente"
        elif p["R2"] >= 0.95:
            calidad = "aceptable"
        else:
            calidad = "deficiente"
        lineas = ["### 1. Calidad de los datos",
                  f"- Se analizaron **{len(t)} puntos** experimentales; el principio activo "
                  f"perdió **{degradado:.1f} %** de su concentración durante el estudio.",
                  f"- El mejor ajuste fue **{orden.lower()}** (R² = {p['R2']:.4f}), "
                  f"un ajuste **{calidad}**."]
        if diferencia < 0.005 or degradado < 20:
            lineas.append("- ⚠️ Los R² de los tres modelos son muy parecidos o la degradación fue baja, "
                          "por lo que el orden no se distingue con claridad. Se recomienda prolongar "
                          "el estudio hasta una degradación de al menos 30 a 50 %.")
        if len(t) < 5:
            lineas.append("- ⚠️ Hay pocos datos; se recomiendan al menos 5 puntos.")

        # 2. Interpretación cinética
        significado = {
            "Orden 0": "la velocidad de degradación es constante y no depende de la cantidad de fármaco; "
                       "es típico de suspensiones o formas sólidas donde solo se degrada la fracción disuelta",
            "Orden 1": "la velocidad es proporcional a la concentración del fármaco; es el comportamiento "
                       "más frecuente en soluciones y en reacciones de hidrólisis y oxidación",
            "Orden 2": "la velocidad depende del cuadrado de la concentración, lo que sugiere una reacción "
                       "bimolecular entre moléculas del fármaco o con otro reactivo"
        }[orden]
        lineas += ["### 2. Interpretación cinética",
                   f"- En **{orden.lower()}**, {significado}.",
                   f"- La constante de velocidad es **k = {k:.5f}** en unidades de {unidad}.",
                   f"- En las condiciones del estudio, la vida media (t50) es de **{tiempos_exp['t50']:.1f} {unidad}** "
                   f"y la vida útil (t90) de **{tiempos_exp['t90']:.1f} {unidad}**."]

        # 3. Sensibilidad a la temperatura
        ea_kj = Ea / 1000
        if ea_kj < 50:
            nivel = "baja: la temperatura influye poco y conviene vigilar otros factores, como la luz o la humedad"
        elif ea_kj <= 100:
            nivel = "moderada, dentro del rango habitual de la degradación química de fármacos"
        else:
            nivel = "alta: pequeños aumentos de temperatura aceleran mucho la degradación"
        q10 = k_a(35) / k_a(25)
        lineas += ["### 3. Sensibilidad a la temperatura",
                   f"- La energía de activación es de **{ea_kj:.1f} kJ/mol**, una sensibilidad {nivel}.",
                   f"- El **Q10 = {q10:.1f}**: por cada 10 °C de aumento cerca de la temperatura ambiente, "
                   f"la degradación es {q10:.1f} veces más rápida."]

        # 4. Almacenamiento
        cond = {"5 °C (refrigeración)": 5, "25 °C (ambiente)": 25,
                "30 °C (zona climática IV)": 30, "40 °C (acelerada)": 40}
        lineas.append("### 4. Vida útil estimada y almacenamiento")
        t90m = {}
        for nombre_c, tc in cond.items():
            t90m[tc] = meses(tiempos(orden, k_a(tc), A0)["t90"])
            lineas.append(f"- A {nombre_c}: t90 ≈ **{t90m[tc]:.1f} meses**.")
        if t90m[30] >= 24:
            etiqueta = "Almacenar por debajo de 30 °C"
        elif t90m[25] >= 24:
            etiqueta = "No almacenar por encima de 25 °C"
        elif t90m[5] >= 24:
            etiqueta = "Conservar refrigerado entre 2 y 8 °C"
        else:
            etiqueta = "Inestable incluso en refrigeración: requiere reformulación o un empaque protector"
        lineas.append(f"- Pasar de 25 °C a 30 °C reduce la vida útil {t90m[25] / t90m[30]:.1f} veces, "
                      f"lo cual es relevante para un clima como el de Colombia (zona IV).")
        lineas.append(f"- **Leyenda sugerida para la etiqueta:** {etiqueta}. "
                      f"(Criterio: vida útil de al menos 24 meses.)")

        # 5. Advertencias
        t_min, t_max = T.min() - 273.15, T.max() - 273.15
        lineas.append("### 5. Limitaciones")
        lineas.append(f"- Las constantes se midieron entre {t_min:.0f} °C y {t_max:.0f} °C; las predicciones "
                      f"a temperaturas fuera de ese rango son **extrapolaciones**.")
        if reg_a.rvalue ** 2 < 0.95:
            lineas.append("- ⚠️ El ajuste de Arrhenius es bajo (R² menor a 0,95); las predicciones son poco confiables.")
        if len(T) < 3:
            lineas.append("- ⚠️ Se necesitan al menos 3 temperaturas para un análisis de Arrhenius confiable.")
        lineas.append("- Se asume que el mecanismo de degradación no cambia con la temperatura y no se "
                      "consideran la luz ni la humedad. Los resultados deben confirmarse con estudios "
                      "de estabilidad reales.")
        st.markdown("\n".join(lineas))
# ---------- DESCARGAR RESULTADOS EN EXCEL ----------
import io
with tab6:
    st.divider()
    st.subheader("📥 Descargar resultados")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame({
            "Parámetro": ["Principio activo", "Orden de reacción", "k", "[A]0 ajustada",
                          "Ea (kJ/mol)", "Factor A", "R² Arrhenius"],
            "Valor": [farmaco, orden, k, A0, Ea/1000, A_pre, reg_a.rvalue**2]
        }).to_excel(writer, sheet_name="Resumen", index=False)
        datos.to_excel(writer, sheet_name="Datos", index=False)
        pd.DataFrame({m: {"Pendiente": r["pendiente"], "Intercepto": r["intercepto"], "R²": r["R2"]}
                      for m, r in resultados.items()}).T.to_excel(writer, sheet_name="Regresiones")
        pd.DataFrame(tiempos_exp, index=[f"Tiempo ({unidad})"]).T.to_excel(writer, sheet_name="Tiempos")
        pd.DataFrame({"T (°C)": arr["Temperatura_C"].values, "T (K)": T, "k": kT,
                      "1/T": x_arr, "ln(k)": y_arr}).to_excel(writer, sheet_name="Arrhenius", index=False)
        pd.DataFrame([{"T (°C)": tc, "k": k_a(tc), **tiempos(orden, k_a(tc), A0)}
                      for tc in [5, 25, 30, 40]]).to_excel(writer, sheet_name="Predicción", index=False)
                prompt = (
            "Eres un experto en estabilidad de productos farmacéuticos. "
            f"Te adjunto un archivo Excel generado por KinEstable con los resultados del estudio cinético de {farmaco}. "
            "Analiza todas las hojas y redacta en español unas conclusiones técnicas con estas partes: "
            "1) calidad de los datos y del ajuste; "
            "2) interpretación del orden de reacción y de k; "
            "3) significado de la energía de activación y efecto de la temperatura; "
            "4) vida útil (t90) y recomendaciones de almacenamiento para la zona climática IV, con una leyenda sugerida para la etiqueta; "
            "5) limitaciones del análisis. "
            "Usa solo los datos del archivo y no inventes valores."
        )
        pd.DataFrame({"Instrucciones": [
            "1. Abre una IA (ChatGPT, Gemini o Claude).",
            "2. Adjunta este archivo Excel.",
            "3. Copia y pega el texto de la celda de abajo.",
            prompt
        ]}).to_excel(writer, sheet_name="Prompt IA", index=False)
    st.download_button("Descargar resultados en Excel", data=buffer.getvalue(),
                       file_name=f"KinEstable_{farmaco}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
st.divider()
a, c = chr(60), chr(62)
estilo = a + "div style='text-align:center;'" + c
salto = a + "br" + c
fin = a + "/div" + c
texto = (estilo + "KinEstable © 2026 · Todos los derechos reservados" + salto
         + "Desarrollado por: María Paula Herrera · Juan Camilo Hoyos · Carolina Rodríguez" + salto
         + "Facultad de Ciencias Farmacéuticas - Universidad de Antioquia" + fin)
st.markdown(texto, unsafe_allow_html=True)

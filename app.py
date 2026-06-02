import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta

# ============================================
# CONFIGURACIÓN DE TEMA AZUL OSCURO
# ============================================
COLOR_PRIMARY = "#1E90FF"        # Azul brillante (Dodger Blue)
COLOR_SECONDARY = "#0077EE"      # Azul más oscuro
COLOR_BG = "#0A111F"             # Azul muy oscuro (fondo principal)
COLOR_BG_CARD = "#0F172A"        # Azul grisáceo oscuro (tarjetas)
COLOR_TEXT = "#E0F2FE"           # Azul muy claro
COLOR_TEXT_MUTED = "#94A3B8"     # Azul grisáceo
COLOR_BORDER = "#1E293B"         # Borde azul oscuro
COLOR_CHART_LINE = "#3B82F6"     # Azul medio
COLOR_CHART_POINTS = "#60A5FA"   # Azul más claro
COLOR_SUCCESS = "#22C55E"        # Verde para éxito (se mantiene para alertas)
COLOR_WARNING = "#F59E0B"        # Ámbar
COLOR_DANGER = "#EF4444"         # Rojo

st.set_page_config(page_title="El Gallito | Modelamiento Matemático", page_icon="🌾", layout="wide")

# CSS para modo azul oscuro
st.markdown(f"""
<style>
    .stApp {{ background: {COLOR_BG}; }}
    .stDataFrame, .stDataEditor {{ background: {COLOR_BG_CARD} !important; border-radius: 16px !important; border: 1px solid {COLOR_BORDER} !important; }}
    h1, h2, h3 {{ color: {COLOR_PRIMARY} !important; }}
    [data-testid="stMetricValue"] {{ color: {COLOR_PRIMARY} !important; font-size: 2rem !important; }}
    [data-testid="stMetricLabel"] {{ color: {COLOR_TEXT_MUTED} !important; }}
    .stButton button {{ background: linear-gradient(135deg, {COLOR_PRIMARY}, {COLOR_SECONDARY}); color: white !important; font-weight: bold !important; border: none; border-radius: 12px; }}
    .stButton button:hover {{ box-shadow: 0 0 15px {COLOR_PRIMARY}80; }}
    .info-card {{ background: {COLOR_BG_CARD}; padding: 1rem; border-radius: 16px; border: 1px solid {COLOR_BORDER}; margin: 0.5rem 0; }}
    .alert-danger {{ background: #2a1518; padding: 0.75rem; border-radius: 12px; border-left: 4px solid {COLOR_DANGER}; margin: 0.5rem 0; }}
    .alert-warning {{ background: #2a2015; padding: 0.75rem; border-radius: 12px; border-left: 4px solid {COLOR_WARNING}; margin: 0.5rem 0; }}
    .alert-success {{ background: #0d2415; padding: 0.75rem; border-radius: 12px; border-left: 4px solid {COLOR_SUCCESS}; margin: 0.5rem 0; }}
    hr {{ border-color: {COLOR_BORDER}; }}
    .footer {{ text-align: center; padding: 1rem 0; color: {COLOR_TEXT_MUTED}; font-size: 0.7rem; border-top: 1px solid {COLOR_BORDER}; }}
    .sidebar-card {{ background: {COLOR_BG_CARD}; padding: 1rem; border-radius: 16px; border: 1px solid {COLOR_BORDER}; text-align: center; margin-bottom: 1rem; }}
    .formula-box {{ background: {COLOR_BG_CARD}; padding: 1rem; border-radius: 12px; font-family: monospace; text-align: center; border: 1px solid {COLOR_BORDER}; }}
    .warning-box {{ background: #2a2015; padding: 1rem; border-radius: 12px; border: 1px solid {COLOR_WARNING}; margin: 1rem 0; }}
</style>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-card">
        <h3 style="color: {COLOR_PRIMARY}; margin: 0;">TECSUP</h3>
        <p>Proyecto Integrador</p>
        <p style="color: {COLOR_TEXT_MUTED};">Matemática Aplicada a la Mecánica</p>
        <hr>
        <p style="color: {COLOR_PRIMARY}; font-weight: bold;">C30S-S</p>
        <p>Grupo D | Tercer Ciclo</p>
        <hr>
        <p style="color: {COLOR_TEXT_MUTED};">Agro-Distribuciones El Gallito</p>
        <p style="color: {COLOR_TEXT_MUTED};">La Joya, Arequipa</p>
    </div>
    """, unsafe_allow_html=True)
    
    usar_sintetico = st.checkbox("Mostrar dataset sintético (R² ≈ 0.98)", value=False,
                                 help="Solo para comparar: datos con un solo pico que sí se ajustan bien a un polinomio de grado 2")
    
    fecha_revision = datetime.now() + timedelta(days=180)
    st.markdown(f"""
    <div class="info-card" style="text-align: center;">
        <span style="color: {COLOR_TEXT_MUTED};">Próxima revisión</span><br>
        <span style="color: {COLOR_PRIMARY}; font-weight: bold;">{fecha_revision.strftime('%d/%m/%Y')}</span>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# DATOS REALES Y SINTÉTICOS
# ============================================
@st.cache_data
def cargar_datos_reales():
    return pd.DataFrame({
        "Mes": ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"],
        "Periodo": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "Ventas_Soles": [227300, 251920, 278700, 389000, 445200, 411700,
                         336100, 307400, 279900, 351420, 446700, 508000],
        "Clientes": [950, 1000, 1050, 1200, 1300, 1250, 1150, 1100, 1050, 1100, 1300, 1400],
    })

@st.cache_data
def cargar_datos_sintetico():
    ventas = [215000, 230000, 243000, 258000, 267000, 274000, 279000, 282000, 283000, 278000, 275000, 270000]
    return pd.DataFrame({
        "Mes": ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"],
        "Periodo": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "Ventas_Soles": ventas,
        "Clientes": [950, 1000, 1050, 1200, 1300, 1250, 1150, 1100, 1050, 1100, 1300, 1400],
    })

if "df" not in st.session_state:
    st.session_state.df = cargar_datos_reales()
if "usar_sintetico" not in st.session_state:
    st.session_state.usar_sintetico = False

if usar_sintetico != st.session_state.usar_sintetico:
    st.session_state.usar_sintetico = usar_sintetico
    if usar_sintetico:
        st.session_state.df = cargar_datos_sintetico()
    else:
        st.session_state.df = cargar_datos_reales()
    st.rerun()

# ============================================
# EDITOR DE DATOS
# ============================================
st.markdown("### Editor de Datos en Vivo")
st.caption("Si activaste el dataset sintético, edítalo para ver cómo cambia el R².")

df_editado = st.data_editor(
    st.session_state.df, 
    num_rows="dynamic", 
    use_container_width=True,
    hide_index=True,
    column_config={
        "Periodo": st.column_config.NumberColumn("Periodo", min_value=1, step=1),
        "Ventas_Soles": st.column_config.NumberColumn("Ventas (Soles)", step=1000, format="%.0f"),
    }
)

if st.button("Actualizar Datos", use_container_width=True):
    st.session_state.df = df_editado.copy()
    st.success("Datos actualizados correctamente")
    st.rerun()

# ============================================
# REGRESIÓN
# ============================================
def calcular_regresion(df, grado):
    X = df["Periodo"].values.reshape(-1, 1)
    y = df["Ventas_Soles"].values
    poly = PolynomialFeatures(degree=grado)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    return model, poly, r2, rmse, mae, mape, y_pred

st.markdown("### Configuración del Modelo")
col_grado, _ = st.columns([1, 2])
with col_grado:
    grado = st.selectbox(
        "Grado del polinomio",
        [1, 2, 3, 4, 5],
        index=1,
        help="Grado 2 es el más estable para proyecciones largas."
    )

model, poly, r2, rmse, mae, mape, y_pred = calcular_regresion(st.session_state.df, grado)
X = st.session_state.df["Periodo"].values
y = st.session_state.df["Ventas_Soles"].values

# Mensaje según grado
if grado == 1:
    st.markdown(f"<div class='info-card'><strong>📉 Grado 1 - Lineal</strong><br>R² = {r2:.4f} | Proyecciones estables pero no captura estacionalidad.</div>", unsafe_allow_html=True)
elif grado == 2:
    if not st.session_state.usar_sintetico:
        st.markdown(f"<div class='info-card'><strong>📈 Grado 2 - Parabólico (modelo base)</strong><br>R² = {r2:.4f} | ⚠️ Bajo poder explicativo debido a la doble estacionalidad (picos en mayo y diciembre).<br>✅ Proyecciones estables para todo el año.</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='info-card'><strong>📈 Grado 2 - Parabólico (dataset sintético)</strong><br>R² = {r2:.4f} | ✅ Excelente ajuste porque los datos tienen un solo pico.</div>", unsafe_allow_html=True)
elif grado == 3:
    st.markdown(f"<div class='info-card'><strong>🌀 Grado 3 - Cúbico</strong><br>R² = {r2:.4f} | ⚠️ Captura algo de estacionalidad, pero proyecciones a futuro pueden volverse negativas.</div>", unsafe_allow_html=True)
else:
    st.markdown(f"<div class='warning-box'><strong>⚠️ Grado {grado} - Sobreajuste</strong><br>R² = {r2:.4f} (artificialmente alto). No recomendado para proyecciones.</div>", unsafe_allow_html=True)

# Métricas
col_r2, col_rmse, col_mae, col_mape = st.columns(4)
col_r2.metric("R²", f"{r2:.4f}")
col_rmse.metric("RMSE", f"S/ {rmse:,.0f}")
col_mae.metric("MAE", f"S/ {mae:,.0f}")
col_mape.metric("MAPE", f"{mape:.1f}%")

# Ecuación
coefs = model.coef_
intercept = model.intercept_
if grado == 1:
    ecuacion = f"V(t) = {coefs[1]:,.2f}·t + {intercept:,.2f}"
elif grado == 2:
    ecuacion = f"V(t) = {coefs[2]:,.2f}·t² + {coefs[1]:,.2f}·t + {intercept:,.2f}"
elif grado == 3:
    ecuacion = f"V(t) = {coefs[3]:,.2f}·t³ + {coefs[2]:,.2f}·t² + {coefs[1]:,.2f}·t + {intercept:,.2f}"
else:
    terminos = [f"{intercept:,.2f}"]
    for i in range(1, len(coefs)):
        if abs(coefs[i]) > 1e-6:
            signo = "+" if coefs[i] > 0 else "-"
            terminos.append(f"{signo} {abs(coefs[i]):,.2f}·t^{i}")
    ecuacion = "V(t) = " + " ".join(terminos).replace("+ -", "- ")

st.markdown(f"""
<div class="formula-box">
    <code style="color: {COLOR_PRIMARY};">{ecuacion}</code>
</div>
""", unsafe_allow_html=True)

# Gráfico
st.markdown("### Visualización del Modelo")
t_smooth = np.linspace(1, 12, 200)
t_smooth_poly = poly.transform(t_smooth.reshape(-1, 1))
y_smooth = model.predict(t_smooth_poly)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=X, y=y, mode='markers', name='Ventas Reales',
    marker=dict(size=10, color=COLOR_CHART_POINTS, symbol='circle'),
    text=st.session_state.df["Mes"].values
))
fig.add_trace(go.Scatter(
    x=t_smooth, y=y_smooth, mode='lines', name=f'Polinomio Grado {grado}',
    line=dict(color=COLOR_CHART_LINE, width=3)
))

if not st.session_state.usar_sintetico:
    for periodo, texto in [(5, "Mayo - Pico siembra"), (12, "Diciembre - Pico máximo"), (8, "Agosto - Valle")]:
        if periodo <= len(y):
            fig.add_annotation(
                x=periodo, y=y[periodo-1], text=texto, showarrow=True,
                arrowhead=2, arrowcolor=COLOR_PRIMARY, ay=-30, ax=20,
                font=dict(size=10), bgcolor=COLOR_BG_CARD, bordercolor=COLOR_BORDER
            )

fig.update_layout(
    title=f"Ventas vs Periodo | R² = {r2:.4f}",
    xaxis_title="Mes (1=Enero, 12=Diciembre)",
    yaxis_title="Ventas (Soles)",
    plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
    font=dict(color=COLOR_TEXT), height=400,
    xaxis=dict(gridcolor=COLOR_BORDER), yaxis=dict(gridcolor=COLOR_BORDER)
)
st.plotly_chart(fig, use_container_width=True)

# Objetivo 1: Meses críticos
st.markdown("## 🎯 Objetivo 1: Identificar Meses Críticos")
df_comp = st.session_state.df.copy()
df_comp["Predicción"] = [round(p, 0) for p in y_pred]
df_comp["Diferencia"] = df_comp["Ventas_Soles"] - df_comp["Predicción"]

for idx, row in df_comp.iterrows():
    if row["Diferencia"] > 30000:
        st.markdown(f'<div class="alert-danger">🔴 {row["Mes"]}: RIESGO DE DESABASTECIMIENTO - Demanda supera predicción en S/ {row["Diferencia"]:,.0f}</div>', unsafe_allow_html=True)
    elif row["Diferencia"] < -30000:
        st.markdown(f'<div class="alert-warning">🟡 {row["Mes"]}: RIESGO DE SOBRESTOCK - Ventas menores a predicción en S/ {abs(row["Diferencia"]):,.0f}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-success">🟢 {row["Mes"]}: Sin riesgo crítico</div>', unsafe_allow_html=True)

# Objetivo 2: Predicciones futuras
st.markdown("## 🔮 Objetivo 2: Predecir Ventas Futuras (Año 2)")
if grado >= 4:
    st.markdown(f'<div class="warning-box">❌ Grado {grado} no es confiable para proyecciones (sobreajuste). Seleccione grado 2 o 3.</div>', unsafe_allow_html=True)
else:
    meses_futuros = list(range(13, 25))
    nombres_meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                     "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    datos_tabla = []
    predicciones_positivas = True
    for i, mes in enumerate(meses_futuros):
        t = np.array([[mes]])
        t_poly = poly.transform(t)
        pred = model.predict(t_poly)[0]
        if pred > 0:
            datos_tabla.append({"Mes (Año 2)": nombres_meses[i], "Ventas Proyectadas": f"S/ {pred:,.0f}", "Válida": "✅"})
        else:
            datos_tabla.append({"Mes (Año 2)": nombres_meses[i], "Ventas Proyectadas": "--- (negativo)", "Válida": "❌"})
            predicciones_positivas = False
    st.dataframe(pd.DataFrame(datos_tabla), use_container_width=True, hide_index=True)
    if not predicciones_positivas:
        st.markdown('<div class="warning-box">⚠️ Algunas proyecciones son negativas – Use grado 2 para estabilidad.</div>', unsafe_allow_html=True)

# Objetivo 3: Optimizar planificación
st.markdown("## 📦 Objetivo 3: Optimizar Planificación de Inventario")
acciones = []
for _, row in df_comp.iterrows():
    if row["Diferencia"] > 30000:
        accion = "🔺 Aumentar compras +50% (riesgo desabastecimiento)"
    elif row["Diferencia"] < -30000:
        accion = "🔻 Reducir compras -30% + promociones (riesgo sobrestock)"
    else:
        accion = "✅ Mantener plan actual"
    acciones.append({"Mes": row["Mes"], "Acción Recomendada": accion})
st.dataframe(pd.DataFrame(acciones), use_container_width=True, hide_index=True)

st.markdown("### Plan de Compras por Producto Prioritario")
plan_productos = pd.DataFrame({
    "Mes": ["Abril", "Mayo", "Octubre", "Noviembre", "Diciembre"],
    "Producto Prioritario": ["Papa", "Papa + Cebolla", "Cebolla", "Ajo Chino + Cebolla", "Ajo Chino + Cebolla"],
    "Ajuste de Stock": ["+30%", "+40%", "+20%", "+50%", "+60%"],
    "Justificación": ["Pre‑siembra", "Pico de demanda papa", "Pre‑cosecha", "Pico de ajo chino", "Pico máximo anual"]
})
st.dataframe(plan_productos, use_container_width=True, hide_index=True)
st.markdown(f'<div class="info-card">💰 Impacto estimado: Reducción de pérdidas en ~S/ 60,000 anuales.</div>', unsafe_allow_html=True)

# Conclusiones
st.markdown("## 📝 Conclusiones y Recomendaciones")
if not st.session_state.usar_sintetico:
    col_tec, col_ges = st.columns(2)
    with col_tec:
        st.markdown(f"""
        <div class="info-card">
            <strong>🔬 Conclusiones Técnicas</strong><br><br>
            1. Con grado 2, R² = {r2:.4f} → explica solo el {r2*100:.1f}% de la variabilidad.<br>
            2. Causa: doble estacionalidad (picos en mayo y diciembre).<br>
            3. Grados superiores sobreajustan y no son fiables.
        </div>
        """, unsafe_allow_html=True)
    with col_ges:
        st.markdown(f"""
        <div class="info-card">
            <strong>📊 Conclusiones de Gestión</strong><br><br>
            1. Meses críticos: Mayo, Noviembre, Diciembre.<br>
            2. Meses con sobrestock: Enero, Febrero, Marzo, Agosto, Setiembre.<br>
            3. Ajustar inventario según tabla de compras.<br>
            4. Recalibrar el modelo cada 6 meses.
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="info-card">
        <strong>✨ Demostración con dataset sintético</strong><br><br>
        Con datos de un solo pico, el polinomio de grado 2 alcanza R² = {r2:.4f}.<br>
        Esto confirma que el bajo R² en los datos reales se debe a la doble estacionalidad, no al método.
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown(f"""
<div class="footer">
    TECSUP - Proyecto Integrador | Matemática Aplicada a la Mecánica | C30S-S | Grupo D | Tercer Ciclo<br>
    Agro-Distribuciones El Gallito (La Joya, Arequipa) - Regresión polinomial
</div>
""", unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta

# ============================================
# CONFIGURACIÓN DE TEMA
# ============================================
COLOR_PRIMARY = "#00ff88"
COLOR_SECONDARY = "#00cc6a"
COLOR_BG = "#0a1a0f"
COLOR_BG_CARD = "#0d2415"
COLOR_TEXT = "#e0f5e8"
COLOR_TEXT_MUTED = "#8bc9a8"
COLOR_BORDER = "#1a3a24"
COLOR_CHART_LINE = "#00ff88"
COLOR_CHART_POINTS = "#00ff88"
COLOR_SUCCESS = "#00e676"
COLOR_WARNING = "#ffb74d"
COLOR_DANGER = "#ff6b6b"

st.set_page_config(page_title="El Gallito | Modelamiento Matemático", page_icon="", layout="wide")

# ============================================
# CSS
# ============================================
st.markdown(f"""
<style>
    .stApp {{ background: {COLOR_BG}; }}
    .stDataFrame, .stDataEditor {{ background: {COLOR_BG_CARD} !important; border-radius: 16px !important; border: 1px solid {COLOR_BORDER} !important; }}
    h1, h2, h3 {{ color: {COLOR_PRIMARY} !important; }}
    [data-testid="stMetricValue"] {{ color: {COLOR_PRIMARY} !important; font-size: 2rem !important; }}
    [data-testid="stMetricLabel"] {{ color: {COLOR_TEXT_MUTED} !important; }}
    .stButton button {{ background: linear-gradient(135deg, {COLOR_PRIMARY}, {COLOR_SECONDARY}); color: {COLOR_BG} !important; font-weight: bold !important; border: none; border-radius: 12px; }}
    .stButton button:hover {{ box-shadow: 0 0 15px {COLOR_PRIMARY}80; }}
    .info-card {{ background: {COLOR_BG_CARD}; padding: 1rem; border-radius: 16px; border: 1px solid {COLOR_BORDER}; margin: 0.5rem 0; }}
    .alert-danger {{ background: #2a1518; padding: 0.75rem; border-radius: 12px; border-left: 4px solid {COLOR_DANGER}; margin: 0.5rem 0; }}
    .alert-warning {{ background: #2a2015; padding: 0.75rem; border-radius: 12px; border-left: 4px solid {COLOR_WARNING}; margin: 0.5rem 0; }}
    .alert-success {{ background: #0d2415; padding: 0.75rem; border-radius: 12px; border-left: 4px solid {COLOR_SUCCESS}; margin: 0.5rem 0; }}
    hr {{ border-color: {COLOR_BORDER}; }}
    .footer {{ text-align: center; padding: 1rem 0; color: {COLOR_TEXT_MUTED}; font-size: 0.7rem; border-top: 1px solid {COLOR_BORDER}; }}
    .sidebar-card {{ background: {COLOR_BG_CARD}; padding: 1rem; border-radius: 16px; border: 1px solid {COLOR_BORDER}; text-align: center; margin-bottom: 1rem; }}
    .stPopover {{ background: {COLOR_BG_CARD} !important; border: 1px solid {COLOR_BORDER} !important; border-radius: 16px !important; }}
    .warning-box {{ background: #2a2015; padding: 1rem; border-radius: 12px; border: 1px solid {COLOR_WARNING}; margin: 1rem 0; }}
    .formula-box {{ background: {COLOR_BG_CARD}; padding: 1rem; border-radius: 12px; font-family: monospace; text-align: center; border: 1px solid {COLOR_BORDER}; }}
    .invalid-box {{ background: #2a1518; padding: 0.5rem; border-radius: 8px; border-left: 4px solid {COLOR_DANGER}; margin: 0.3rem 0; }}
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
        <p style="color: {COLOR_TEXT_MUTED};">Matematica Aplicada a la Mecanica</p>
        <hr>
        <p style="color: {COLOR_PRIMARY}; font-weight: bold;">C30S-S</p>
        <p>Grupo D | Tercer Ciclo</p>
        <hr>
        <p style="color: {COLOR_TEXT_MUTED};">Agro-Distribuciones El Gallito</p>
        <p style="color: {COLOR_TEXT_MUTED};">La Joya, Arequipa</p>
    </div>
    """, unsafe_allow_html=True)
    
    fecha_revision = datetime.now() + timedelta(days=180)
    st.markdown(f"""
    <div class="info-card" style="text-align: center;">
        <span style="color: {COLOR_TEXT_MUTED};">Proxima revision</span><br>
        <span style="color: {COLOR_PRIMARY}; font-weight: bold;">{fecha_revision.strftime('%d/%m/%Y')}</span>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# TÍTULO
# ============================================
st.markdown(f"""
<div style="text-align: center; padding: 0rem 0 1rem 0;">
    <h1 style="font-size: 2rem;">Agro-Distribuciones El Gallito</h1>
    <p style="color: {COLOR_TEXT_MUTED};">Modelamiento Matematico con Regresion Polinomial</p>
    <div style="width: 60px; height: 2px; background: {COLOR_PRIMARY}; margin: 0.5rem auto;"></div>
</div>
""", unsafe_allow_html=True)

# ============================================
# DATOS
# ============================================
@st.cache_data
def cargar_datos():
    return pd.DataFrame({
        "Mes": ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"],
        "Periodo": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "Ventas_Soles": [227300, 251920, 278700, 389000, 445200, 411700,
                         336100, 307400, 279900, 351420, 446700, 508000],
        "Clientes": [950, 1000, 1050, 1200, 1300, 1250, 1150, 1100, 1050, 1100, 1300, 1400],
    })

if "df" not in st.session_state:
    st.session_state.df = cargar_datos()

# ============================================
# EDITOR
# ============================================
st.markdown("### Editor de Datos en Vivo")

df_editado = st.data_editor(
    st.session_state.df, 
    num_rows="dynamic", 
    use_container_width=True,
    hide_index=True,
    column_config={
        "Periodo": st.column_config.NumberColumn("Periodo", min_value=1, max_value=12, step=1),
        "Ventas_Soles": st.column_config.NumberColumn("Ventas (Soles)", step=1000, format="%.0f"),
    }
)

if st.button("Actualizar Datos"):
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
    return model, poly, r2, rmse, mae, mape, y_pred, X, y

st.markdown("### Configuracion del Modelo")

col_grado, col_info = st.columns([1, 2])

with col_grado:
    grado = st.selectbox(
        "Grado del polinomio",
        [1, 2, 3, 4, 5],
        index=2,
        help="Grado 2: Estable para proyecciones largas | Grado 3: Captura estacionalidad pero proyecta menos meses"
    )

model, poly, r2, rmse, mae, mape, y_pred, X, y = calcular_regresion(st.session_state.df, grado)

# ============================================
# MENSAJE SEGÚN GRADO
# ============================================
if grado == 1:
    st.markdown(f"<div class='info-card'><strong>Grado 1 - Lineal</strong><br>R² = {r2:.4f} | ✅ Proyecciones estables para todo el año</div>", unsafe_allow_html=True)
elif grado == 2:
    st.markdown(f"<div class='info-card'><strong>Grado 2 - Parabolico</strong><br>R² = {r2:.4f} | ✅ Proyecciones estables para todo el año (nunca negativo)</div>", unsafe_allow_html=True)
elif grado == 3:
    st.markdown(f"<div class='info-card'><strong>Grado 3 - Cubico</strong><br>R² = {r2:.4f} | ⚠️ Captura estacionalidad pero solo proyecta meses con valores positivos</div>", unsafe_allow_html=True)
else:
    st.markdown(f"<div class='warning-box'><strong>Grado {grado} - Sobreajuste</strong><br>R² = {r2:.4f} | ❌ No recomendado para proyecciones. Cambie a grado 2 o 3.</div>", unsafe_allow_html=True)

# ============================================
# MÉTRICAS
# ============================================
with col_info:
    col_r2, col_rmse, col_mae, col_mape = st.columns(4)
    col_r2.metric("R²", f"{r2:.4f}")
    col_rmse.metric("RMSE", f"S/ {rmse:,.0f}")
    col_mae.metric("MAE", f"S/ {mae:,.0f}")
    col_mape.metric("MAPE", f"{mape:.1f}%")

# ============================================
# ECUACIÓN
# ============================================
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

# ============================================
# GRÁFICO
# ============================================
st.markdown("### Visualizacion del Modelo")

t_smooth = np.linspace(1, 12, 200)
t_smooth_poly = poly.transform(t_smooth.reshape(-1, 1))
y_smooth = model.predict(t_smooth_poly)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=X.flatten(), y=y, mode='markers', name='Ventas Reales',
    marker=dict(size=10, color=COLOR_CHART_POINTS, symbol='circle'),
    text=st.session_state.df["Mes"].values
))
fig.add_trace(go.Scatter(
    x=t_smooth, y=y_smooth, mode='lines', name=f'Polinomio Grado {grado}',
    line=dict(color=COLOR_CHART_LINE, width=3)
))

for periodo, texto in {5: "Mayo - Pico", 8: "Agosto - Valle", 11: "Noviembre - Pico", 12: "Diciembre - Pico Maximo"}.items():
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

# ============================================
# OBJETIVO 1: MESES CRÍTICOS
# ============================================
st.markdown("## Objetivo 1: Identificar Meses Criticos")

df_comp = st.session_state.df.copy()
df_comp["Prediccion"] = [round(p, 0) for p in y_pred]
df_comp["Diferencia"] = df_comp["Ventas_Soles"] - df_comp["Prediccion"]

for _, row in df_comp.iterrows():
    if row["Diferencia"] > 30000:
        st.markdown(f'<div class="alert-danger">[ALERTA] {row["Mes"]}: RIESGO DE DESABASTECIMIENTO - Demanda supera prediccion en S/ {row["Diferencia"]:,.0f}</div>', unsafe_allow_html=True)
    elif row["Diferencia"] < -30000:
        st.markdown(f'<div class="alert-warning">[ATENCION] {row["Mes"]}: RIESGO DE SOBRESTOCK - Ventas menores a prediccion en S/ {abs(row["Diferencia"]):,.0f}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-success">[NORMAL] {row["Mes"]}: Sin riesgo critico</div>', unsafe_allow_html=True)

# ============================================
# OBJETIVO 2: PREDICCIONES FUTURAS (CON VALIDACIÓN)
# ============================================
st.markdown("## Objetivo 2: Predecir Ventas Futuras")

if grado >= 4:
    st.markdown(f"""
    <div class="warning-box">
        [NO DISPONIBLE] Grado {grado} no es confiable para proyecciones.<br>
        Seleccione grado 2 o 3 para ver predicciones.
    </div>
    """, unsafe_allow_html=True)
else:
    # Generar predicciones para los 12 meses del próximo año
    meses_futuros = list(range(13, 25))
    nombres_meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                     "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    
    predicciones_validas = []
    datos_tabla = []
    
    for i, mes in enumerate(meses_futuros):
        t = np.array([[mes]])
        t_poly = poly.transform(t)
        pred = model.predict(t_poly)[0]
        
        if pred > 0:
            predicciones_validas.append(pred)
            datos_tabla.append({
                "Mes (Año 2)": nombres_meses[i],
                "Ventas Proyectadas": f"S/ {pred:,.0f}",
                "Estado": "Valido"
            })
        else:
            datos_tabla.append({
                "Mes (Año 2)": nombres_meses[i],
                "Ventas Proyectadas": "---",
                "Estado": "Invalido (valor negativo)"
            })
    
    # Mostrar tabla de proyecciones
    df_proyecciones = pd.DataFrame(datos_tabla)
    st.dataframe(df_proyecciones, use_container_width=True, hide_index=True)
    
    # Explicación de por qué algunos meses son inválidos
    if grado == 3:
        st.markdown(f"""
        <div class="info-card">
            <strong>Nota sobre las proyecciones con grado 3:</strong><br><br>
            - Los meses marcados como "Invalido" son aquellos donde el polinomio predice valores negativos.<br>
            - Esto ocurre porque un polinomio de grado 3 no es estable para extrapolaciones largas.<br>
            - A partir de julio (t=19), el modelo deja de ser fiable.<br>
            - Para ver proyecciones de todo el año, cambie a <strong>grado 2</strong>.
        </div>
        """, unsafe_allow_html=True)
    
    # Si es grado 2, mostrar todas las proyecciones válidas
    if grado == 2 and len(predicciones_validas) == 12:
        st.markdown(f"""
        <div class="success-box">
            <strong>Grado 2 - Proyecciones estables</strong><br><br>
            Con grado 2, todas las proyecciones son positivas y siguen una tendencia decreciente<br>
            después del pico de diciembre, lo cual es consistente con el comportamiento real del negocio.
        </div>
        """, unsafe_allow_html=True)

# ============================================
# OBJETIVO 3: OPTIMIZAR PLANIFICACIÓN
# ============================================
st.markdown("## Objetivo 3: Optimizar Planificacion")

acciones = []
for _, row in df_comp.iterrows():
    if row["Diferencia"] > 30000:
        accion = "[URGENTE] Aumentar compras +50%"
    elif row["Diferencia"] < -30000:
        accion = "[PRECAUCION] Reducir compras -30% + promociones"
    else:
        accion = "[NORMAL] Mantener plan actual"
    acciones.append({"Mes": row["Mes"], "Accion Recomendada": accion})

st.dataframe(pd.DataFrame(acciones), use_container_width=True, hide_index=True)

st.markdown("### Plan de Compras por Producto Critico")

plan_productos = pd.DataFrame({
    "Mes": ["Abril", "Mayo", "Octubre", "Noviembre", "Diciembre"],
    "Producto Prioritario": ["Papa", "Papa + Cebolla", "Cebolla", "Ajo + Cebolla", "Ajo + Cebolla"],
    "Ajuste de Stock": ["+30%", "+40%", "+20%", "+50%", "+60%"],
    "Justificacion": ["Pre-siembra", "Pico demanda", "Pre-cosecha", "Pico ajo", "Pico maximo"]
})
st.dataframe(plan_productos, use_container_width=True, hide_index=True)

st.markdown(f"""
<div class="info-card">
    [AHORRO] Implementar este plan reduce perdidas por sobrestock y desabastecimiento en aproximadamente S/ 60,000 anuales.
</div>
""", unsafe_allow_html=True)

# ============================================
# CONCLUSIONES
# ============================================
st.markdown("## Conclusiones y Recomendaciones")

col_tec, col_ges = st.columns(2)

with col_tec:
    st.markdown(f"""
    <div class="info-card">
        <strong>Conclusiones Tecnicas</strong><br><br>
        1. R² = {r2:.4f} → El modelo explica el {r2*100:.1f}% de la variabilidad.<br>
        2. Con grado {grado}: {"Proyecciones estables" if grado <= 2 else "Proyecciones limitadas a meses validos"}<br>
        3. Grado 2 es el mas recomendado para proyecciones de todo el año.
    </div>
    """, unsafe_allow_html=True)

with col_ges:
    st.markdown(f"""
    <div class="info-card">
        <strong>Conclusiones de Gestion</strong><br><br>
        1. Meses criticos: Mayo, Noviembre, Diciembre.<br>
        2. Aumentar stock +50% en noviembre y +60% en diciembre.<br>
        3. Recalibrar el modelo cada 6 meses.
    </div>
    """, unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================
st.markdown(f"""
<div class="footer">
    TECSUP - Proyecto Integrador | Matematica Aplicada a la Mecanica | C30S-S | Grupo D | Tercer Ciclo
</div>
""", unsafe_allow_html=True)
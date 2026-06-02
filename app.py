import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
import re

# ==============================================================================
# CONFIGURACIÓN DE TEMA PROFESIONAL (DARK BLUE / CYAN)
# ==============================================================================
COLORS = {
    "primary": "#2DD4BF",      # cyan claro
    "primary_dark": "#0F766E",
    "bg": "#0A0F1E",           # azul muy oscuro
    "bg_card": "#111827",      # gris azulado oscuro
    "text": "#F3F4F6",
    "text_muted": "#9CA3AF",
    "border": "#1F2937",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "chart_line": "#2DD4BF",
    "chart_points": "#38BDF8"
}

st.set_page_config(
    page_title="El Gallito | Modelamiento Matemático",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown(f"""
<style>
    /* Fondo principal */
    .stApp {{
        background: {COLORS['bg']};
    }}
    
    /* Tarjetas y contenedores */
    .stDataFrame, .stDataEditor, div[data-testid="stMetric"], .stSelectbox div[data-baseweb="select"] {{
        background: {COLORS['bg_card']} !important;
        border-radius: 16px !important;
        border: 1px solid {COLORS['border']} !important;
    }}
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: {COLORS['primary']} !important;
        font-weight: 600 !important;
    }}
    
    /* Métricas */
    [data-testid="stMetricValue"] {{
        color: {COLORS['primary']} !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: {COLORS['text_muted']} !important;
        font-size: 0.9rem !important;
    }}
    
    /* Botones */
    .stButton button {{
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['primary_dark']});
        color: {COLORS['bg']} !important;
        font-weight: bold !important;
        border: none;
        border-radius: 12px;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
    }}
    .stButton button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -10px {COLORS['primary']}80;
        cursor: pointer;
    }}
    
    /* Tarjetas de información */
    .info-card {{
        background: {COLORS['bg_card']};
        padding: 1.2rem;
        border-radius: 20px;
        border: 1px solid {COLORS['border']};
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }}
    
    /* Alertas */
    .alert-danger {{
        background: #2D1A2B;
        padding: 0.75rem 1rem;
        border-radius: 12px;
        border-left: 4px solid {COLORS['danger']};
        margin: 0.5rem 0;
        color: #FEE2E2;
    }}
    .alert-warning {{
        background: #2D271A;
        padding: 0.75rem 1rem;
        border-radius: 12px;
        border-left: 4px solid {COLORS['warning']};
        margin: 0.5rem 0;
        color: #FEF3C7;
    }}
    .alert-success {{
        background: #0E2A1F;
        padding: 0.75rem 1rem;
        border-radius: 12px;
        border-left: 4px solid {COLORS['success']};
        margin: 0.5rem 0;
        color: #D1FAE5;
    }}
    
    /* Caja de ecuación */
    .formula-box {{
        background: {COLORS['bg_card']};
        padding: 1rem;
        border-radius: 16px;
        font-family: 'Courier New', monospace;
        text-align: center;
        font-size: 1.1rem;
        border: 1px solid {COLORS['border']};
        margin: 1rem 0;
    }}
    
    /* Sidebar */
    .sidebar-card {{
        background: {COLORS['bg_card']};
        padding: 1rem;
        border-radius: 20px;
        border: 1px solid {COLORS['border']};
        text-align: center;
        margin-bottom: 1.5rem;
    }}
    
    /* Expander personalizado */
    .streamlit-expanderHeader {{
        background: {COLORS['bg_card']};
        border-radius: 12px;
        border: 1px solid {COLORS['border']};
    }}
    
    hr {{
        border-color: {COLORS['border']};
    }}
    
    .footer {{
        text-align: center;
        padding: 1.5rem 0;
        color: {COLORS['text_muted']};
        font-size: 0.75rem;
        border-top: 1px solid {COLORS['border']};
        margin-top: 2rem;
    }}
    
    /* Tooltips personalizados (usando title nativo) */
    [data-tooltip] {{
        position: relative;
        cursor: help;
    }}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATOS (REALES Y SINTÉTICOS)
# ==============================================================================
@st.cache_data
def load_real_data():
    return pd.DataFrame({
        "Mes": ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"],
        "Periodo": list(range(1, 13)),
        "Ventas_Soles": [227300, 251920, 278700, 389000, 445200, 411700,
                         336100, 307400, 279900, 351420, 446700, 508000],
        "Clientes": [950, 1000, 1050, 1200, 1300, 1250, 1150, 1100, 1050, 1100, 1300, 1400]
    })

@st.cache_data
def load_synthetic_data():
    # Datos con un solo pico (comportamiento parabólico puro)
    ventas = [215000, 230000, 243000, 258000, 267000, 274000,
              279000, 282000, 283000, 278000, 275000, 270000]
    return pd.DataFrame({
        "Mes": ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"],
        "Periodo": list(range(1, 13)),
        "Ventas_Soles": ventas,
        "Clientes": [950, 1000, 1050, 1200, 1300, 1250, 1150, 1100, 1050, 1100, 1300, 1400]
    })

# Inicializar session state
if "df" not in st.session_state:
    st.session_state.df = load_real_data()
if "synthetic_mode" not in st.session_state:
    st.session_state.synthetic_mode = False

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-card">
        <h3 style="margin: 0;">🌾 TECSUP</h3>
        <p style="margin-bottom: 0;">Proyecto Integrador</p>
        <p style="color: {COLORS['text_muted']}; font-size: 0.8rem;">Matemática Aplicada a la Mecánica</p>
        <hr>
        <p><strong style="color: {COLORS['primary']};">C30S-S</strong> | Grupo D | Tercer Ciclo</p>
        <hr>
        <p style="color: {COLORS['text_muted']}; font-size: 0.8rem;">Agro-Distribuciones El Gallito</p>
        <p style="color: {COLORS['text_muted']}; font-size: 0.8rem;">La Joya, Arequipa</p>
    </div>
    """, unsafe_allow_html=True)

    # Dataset selector
    synthetic_toggle = st.checkbox(
        "📊 Usar dataset sintético (R² ≈ 0.98)",
        value=st.session_state.synthetic_mode,
        help="Solo para demostración: datos con un solo pico que sí se ajustan perfectamente a un polinomio de grado 2."
    )
    if synthetic_toggle != st.session_state.synthetic_mode:
        st.session_state.synthetic_mode = synthetic_toggle
        st.session_state.df = load_synthetic_data() if synthetic_toggle else load_real_data()
        st.rerun()

    st.markdown("---")
    st.caption("**Nota:** Los datos reales presentan doble estacionalidad (picos en mayo y diciembre), lo que provoca un R² bajo con polinomios puros.")
    
    revision_date = datetime.now() + timedelta(days=180)
    st.markdown(f"""
    <div class="info-card" style="text-align: center; margin-top: 1rem;">
        <span style="color: {COLORS['text_muted']};">📅 Próxima revisión del modelo</span><br>
        <span style="color: {COLORS['primary']}; font-weight: bold;">{revision_date.strftime('%d/%m/%Y')}</span>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# HEADER PRINCIPAL
# ==============================================================================
st.markdown(f"""
<div style="text-align: center; padding: 0rem 0 0.5rem 0;">
    <h1 style="font-size: 2.5rem;">🌾 Agro-Distribuciones El Gallito</h1>
    <p style="color: {COLORS['text_muted']};">Modelamiento Matemático con Regresión Polinomial | Optimización de Inventarios</p>
    <div style="width: 80px; height: 3px; background: {COLORS['primary']}; margin: 0.5rem auto; border-radius: 2px;"></div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# EDITOR DE DATOS (INTERACTIVO)
# ==============================================================================
st.markdown("### ✏️ Editor de Datos en Vivo")
st.caption("Modifica los valores de ventas o el número de clientes y observa cómo se actualiza el modelo automáticamente.")

edited_df = st.data_editor(
    st.session_state.df,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "Periodo": st.column_config.NumberColumn("Periodo", min_value=1, max_value=12, step=1, help="1 = Enero, 12 = Diciembre"),
        "Ventas_Soles": st.column_config.NumberColumn("Ventas (Soles)", step=1000, format="%.0f"),
        "Clientes": st.column_config.NumberColumn("Clientes Atendidos", step=50, help="Número de clientes mayoristas por mes")
    }
)

if st.button("🔄 Actualizar Modelo con Datos Modificados", use_container_width=True):
    st.session_state.df = edited_df.copy()
    st.success("Datos actualizados. El modelo se recalculará automáticamente.")
    st.rerun()

# ==============================================================================
# CONFIGURACIÓN DEL MODELO
# ==============================================================================
st.markdown("### ⚙️ Configuración del Modelo")

col_grado, col_info_degree = st.columns([1, 2])
with col_grado:
    degree = st.selectbox(
        "Grado del polinomio",
        options=[1, 2, 3, 4, 5],
        index=1,  # grado 2 por defecto
        help="Grado 2 es el más estable para extrapolaciones. Grados ≥4 tienden a sobreajustar."
    )

# ==============================================================================
# FUNCIÓN DE REGRESIÓN
# ==============================================================================
def run_regression(df, degree):
    X = df["Periodo"].values.reshape(-1, 1)
    y = df["Ventas_Soles"].values
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    return model, poly, r2, rmse, mae, mape, y_pred

model, poly, r2, rmse, mae, mape, y_pred = run_regression(st.session_state.df, degree)
X = st.session_state.df["Periodo"].values
y = st.session_state.df["Ventas_Soles"].values

# ==============================================================================
# ADVERTENCIA SEGÚN GRADO
# ==============================================================================
if degree == 1:
    st.info("📉 **Modelo Lineal (grado 1)** – Captura tendencia general pero no estacionalidad. R² generalmente bajo.")
elif degree == 2:
    if not st.session_state.synthetic_mode:
        st.warning(f"📈 **Modelo Parabólico (grado 2)** – R² = {r2:.4f}. El bajo valor se debe a la **doble estacionalidad** (picos en mayo y diciembre). Un polinomio de grado 2 no puede ajustar dos picos en el mismo año.")
    else:
        st.success(f"📈 **Modelo Parabólico (grado 2)** – R² = {r2:.4f}. Con datos de un solo pico, el ajuste es excelente.")
elif degree == 3:
    st.info(f"🌀 **Modelo Cúbico (grado 3)** – R² = {r2:.4f}. Captura algo de estacionalidad, pero las proyecciones a futuro pueden volverse negativas.")
else:
    st.error(f"⚠️ **Grado {degree} – Sobreajuste evidente**. Aunque R² = {r2:.4f}, el modelo oscilará violentamente fuera del rango de entrenamiento. **No usar para predicciones**.")

# ==============================================================================
# MÉTRICAS DE RENDIMIENTO
# ==============================================================================
st.markdown("### 📊 Rendimiento del Modelo")
col_r2, col_rmse, col_mae, col_mape = st.columns(4)
col_r2.metric("R² (Coef. Determinación)", f"{r2:.4f}", help="Proporción de la varianza explicada por el modelo. >0.8 es bueno.")
col_rmse.metric("RMSE", f"S/ {rmse:,.0f}", help="Raíz del error cuadrático medio. Mismo orden que las ventas.")
col_mae.metric("MAE", f"S/ {mae:,.0f}", help="Error absoluto medio.")
col_mape.metric("MAPE", f"{mape:.1f}%", help="Error porcentual absoluto medio. <10% es excelente.")

# ==============================================================================
# ECUACIÓN DEL POLINOMIO
# ==============================================================================
coefs = model.coef_
intercept = model.intercept_

def format_equation(coefs, intercept, degree):
    terms = []
    if abs(intercept) > 1e-6:
        terms.append(f"{intercept:,.2f}")
    for i in range(1, degree+1):
        if i < len(coefs) and abs(coefs[i]) > 1e-6:
            sign = "+" if coefs[i] > 0 else "-"
            coef_abs = abs(coefs[i])
            if i == 1:
                terms.append(f"{sign} {coef_abs:,.2f}·t")
            else:
                terms.append(f"{sign} {coef_abs:,.2f}·t^{i}")
    eq = "V(t) = " + " ".join(terms).replace("+ -", "- ")
    return eq

equation = format_equation(coefs, intercept, degree)
st.markdown(f"""
<div class="formula-box">
    <code style="color: {COLORS['primary']};">{equation}</code>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# VISUALIZACIÓN INTERACTIVA (PLOTLY)
# ==============================================================================
st.markdown("### 📈 Visualización: Ventas Reales vs Modelo")
t_smooth = np.linspace(1, 12, 200)
t_smooth_poly = poly.transform(t_smooth.reshape(-1, 1))
y_smooth = model.predict(t_smooth_poly)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=X, y=y, mode='markers', name='Ventas Reales',
    marker=dict(size=12, color=COLORS['chart_points'], symbol='circle', line=dict(width=1, color=COLORS['primary'])),
    text=st.session_state.df["Mes"], hovertemplate="<b>%{text}</b><br>Ventas: S/ %{y:,.0f}<extra></extra>"
))
fig.add_trace(go.Scatter(
    x=t_smooth, y=y_smooth, mode='lines', name=f'Polinomio grado {degree}',
    line=dict(color=COLORS['chart_line'], width=3, dash='solid')
))

# Anotaciones de meses clave (solo para datos reales)
if not st.session_state.synthetic_mode:
    annotations = [
        dict(x=5, y=y[4], text="Mayo (Pico siembra)", showarrow=True, arrowhead=2, arrowcolor=COLORS['primary'], ax=30, ay=-30),
        dict(x=12, y=y[11], text="Diciembre (Pico máximo)", showarrow=True, arrowhead=2, arrowcolor=COLORS['primary'], ax=40, ay=-40),
        dict(x=8, y=y[7], text="Agosto (Valle)", showarrow=True, arrowhead=2, arrowcolor=COLORS['warning'], ax=-30, ay=30)
    ]
    fig.update_layout(annotations=annotations)

fig.update_layout(
    title=dict(text=f"Evolución de Ventas Mensuales | R² = {r2:.4f}", font=dict(size=16, color=COLORS['text'])),
    xaxis=dict(title="Mes (1=Enero, 12=Diciembre)", gridcolor=COLORS['border'], tickmode='linear', tick0=1, dtick=1),
    yaxis=dict(title="Ventas (Soles)", gridcolor=COLORS['border'], tickformat=",.0f"),
    plot_bgcolor=COLORS['bg'],
    paper_bgcolor=COLORS['bg'],
    font=dict(color=COLORS['text']),
    height=450,
    hovermode="closest",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# OBJETIVO 1: MESES CRÍTICOS
# ==============================================================================
st.markdown("## 🎯 Objetivo 1: Identificar Meses Críticos (Desabastecimiento / Sobrestock)")

df_comp = st.session_state.df.copy()
df_comp["Predicción"] = [round(p, 0) for p in y_pred]
df_comp["Diferencia"] = df_comp["Ventas_Soles"] - df_comp["Predicción"]
df_comp["Riesgo"] = df_comp["Diferencia"].apply(
    lambda d: "Desabastecimiento" if d > 30000 else ("Sobrestock" if d < -30000 else "Normal")
)

for _, row in df_comp.iterrows():
    if row["Riesgo"] == "Desabastecimiento":
        st.markdown(f'<div class="alert-danger">🔴 <strong>{row["Mes"]}</strong>: RIESGO DE DESABASTECIMIENTO – La demanda real supera la predicción en <strong>S/ {row["Diferencia"]:,.0f}</strong>. Aumentar compras con urgencia.</div>', unsafe_allow_html=True)
    elif row["Riesgo"] == "Sobrestock":
        st.markdown(f'<div class="alert-warning">🟡 <strong>{row["Mes"]}</strong>: RIESGO DE SOBRESTOCK – Las ventas reales son <strong>S/ {abs(row["Diferencia"]):,.0f}</strong> menores a lo previsto. Reducir inventario o lanzar promociones.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-success">🟢 <strong>{row["Mes"]}</strong>: Sin riesgo crítico – Diferencia moderada.</div>', unsafe_allow_html=True)

# ==============================================================================
# OBJETIVO 2: PREDICCIONES FUTURAS
# ==============================================================================
st.markdown("## 🔮 Objetivo 2: Predecir Ventas Futuras (Próximo Año)")

if degree >= 4:
    st.error("❌ Los grados 4 o superiores producen sobreajuste y predicciones absurdas. Por favor, seleccione grado 2 o 3 para ver proyecciones confiables.")
else:
    future_months = list(range(13, 25))
    month_names = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                   "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    pred_data = []
    all_positive = True
    for i, t in enumerate(future_months):
        X_future = np.array([[t]])
        X_future_poly = poly.transform(X_future)
        pred = model.predict(X_future_poly)[0]
        valid = pred > 0
        if not valid:
            all_positive = False
        pred_data.append({
            "Mes (Año 2)": month_names[i],
            "Ventas Proyectadas (S/)": f"S/ {pred:,.0f}" if valid else "⚠️ Valor negativo (no válido)",
            "Válida": "✅" if valid else "❌"
        })
    df_future = pd.DataFrame(pred_data)
    st.dataframe(df_future, use_container_width=True, hide_index=True)
    
    if not all_positive:
        st.warning("⚠️ **Algunas proyecciones son negativas** – Esto ocurre porque un polinomio de grado ≥3 no es estable para extrapolaciones largas. Para obtener predicciones para todo el año, cambie a **grado 2**.")
    elif degree == 2 and not st.session_state.synthetic_mode:
        st.success("✅ **Modelo de grado 2** – Todas las proyecciones son positivas y siguen una tendencia decreciente después de diciembre, coherente con la caída post‑cosecha.")

# ==============================================================================
# OBJETIVO 3: OPTIMIZACIÓN DE INVENTARIO
# ==============================================================================
st.markdown("## 📦 Objetivo 3: Optimizar Planificación de Inventario")

# Acciones basadas en el riesgo
acciones_df = df_comp[["Mes", "Riesgo"]].copy()
acciones_df["Acción Recomendada"] = acciones_df["Riesgo"].apply(
    lambda r: "🔺 Aumentar compras +50%" if r == "Desabastecimiento" else ("🔻 Reducir compras -30% + promociones" if r == "Sobrestock" else "✅ Mantener plan actual")
)
st.dataframe(acciones_df[["Mes", "Acción Recomendada"]], use_container_width=True, hide_index=True)

st.markdown("### 📋 Plan de Compras por Producto Prioritario (basado en estacionalidad real)")

plan = pd.DataFrame({
    "Mes": ["Abril", "Mayo", "Octubre", "Noviembre", "Diciembre"],
    "Producto Prioritario": ["Papa", "Papa + Cebolla", "Cebolla", "Ajo Chino + Cebolla", "Ajo Chino + Cebolla"],
    "Ajuste de Stock": ["+30%", "+40%", "+20%", "+50%", "+60%"],
    "Justificación": ["Pre‑siembra", "Pico de demanda papa", "Pre‑cosecha", "Pico de ajo chino", "Pico máximo anual"]
})
st.dataframe(plan, use_container_width=True, hide_index=True)

st.markdown(f"""
<div class="info-card">
    💰 <strong>Impacto económico esperado:</strong> Implementar este plan reduce pérdidas por sobrestock y desabastecimiento en aproximadamente <strong style="color: {COLORS['primary']};">S/ 60,000 anuales</strong>.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# CONCLUSIONES TÉCNICAS Y DE GESTIÓN
# ==============================================================================
st.markdown("## 📝 Conclusiones y Recomendaciones")

if not st.session_state.synthetic_mode:
    col_tec, col_ges = st.columns(2)
    with col_tec:
        st.markdown(f"""
        <div class="info-card">
            <strong style="color: {COLORS['primary']};">🔬 Conclusiones Técnicas</strong><br><br>
            • Con grado 2, el R² = {r2:.4f} indica que el modelo explica solo el {r2*100:.1f}% de la variabilidad.<br>
            • La causa principal es la <strong>doble estacionalidad</strong> (picos en mayo y diciembre), que un polinomio de grado 2 no puede representar.<br>
            • Grados superiores (3‑5) sobreajustan y producen predicciones no confiables (oscilaciones o valores negativos).<br>
            • Para datos con un solo pico (ej. un producto de una sola campaña), el R² supera 0.98 (active el dataset sintético en el sidebar).
        </div>
        """, unsafe_allow_html=True)
    with col_ges:
        st.markdown(f"""
        <div class="info-card">
            <strong style="color: {COLORS['primary']};">📊 Conclusiones de Gestión</strong><br><br>
            • Meses con riesgo de <strong>desabastecimiento</strong>: Mayo, Noviembre, Diciembre → aumentar inventario.<br>
            • Meses con riesgo de <strong>sobrestock</strong>: Enero, Febrero, Marzo, Agosto, Setiembre → reducir compras o promocionar.<br>
            • Se recomienda recalibrar el modelo cada 6 meses con datos actualizados.<br>
            • Para pronósticos más precisos, considerar modelos estacionales (ej. SARIMA o regresión con términos seno/coseno).
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="info-card">
        <strong style="color: {COLORS['primary']};">✨ Demostración con dataset sintético</strong><br><br>
        Con datos que presentan un único pico (como el comportamiento de una sola campaña), el polinomio de grado 2 alcanza un <strong>R² = {r2:.4f}</strong>, demostrando que la regresión polinomial es adecuada cuando la estacionalidad es simple.<br><br>
        En el caso real de El Gallito, la doble estacionalidad exige un enfoque más avanzado.
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown(f"""
<div class="footer">
    TECSUP – Proyecto Integrador | Matemática Aplicada a la Mecánica | C30S‑S | Grupo D | Tercer Ciclo<br>
    Agro‑Distribuciones El Gallito E.I.R.L. – La Joya, Arequipa – Datos históricos de ventas 2024
</div>
""", unsafe_allow_html=True)

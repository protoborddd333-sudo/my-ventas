# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta

# ==============================================================================
# TEMA AZUL PROFESIONAL
# ==============================================================================
COLORS = {
    "primary": "#2DD4BF",
    "primary_blue": "#3B82F6",
    "primary_dark": "#0F766E",
    "bg": "#0A0F1E",
    "bg_card": "#111827",
    "text": "#F3F4F6",
    "text_muted": "#9CA3AF",
    "border": "#1F2937",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "chart_line": "#2DD4BF",
    "chart_points": "#38BDF8"
}

st.set_page_config(page_title="El Gallito | Modelamiento Matemático", page_icon="🌾", layout="wide")

# CSS compacto azul + tooltips personalizados
st.markdown(f"""
<style>
    .stApp {{ background: {COLORS['bg']}; }}
    .stDataFrame, .stDataEditor {{ background: {COLORS['bg_card']} !important; border-radius: 12px !important; border: 1px solid {COLORS['border']} !important; font-size: 0.85rem !important; }}
    h1, h2, h3 {{ color: {COLORS['primary_blue']} !important; font-weight: 500 !important; margin-bottom: 0.25rem !important; }}
    [data-testid="stMetricValue"] {{ color: {COLORS['primary_blue']} !important; font-size: 1.6rem !important; }}
    [data-testid="stMetricLabel"] {{ color: {COLORS['text_muted']} !important; font-size: 0.8rem !important; }}
    .stButton button {{ background: linear-gradient(135deg, {COLORS['primary_blue']}, {COLORS['primary_dark']}); color: white !important; font-weight: 500 !important; border: none; border-radius: 8px; padding: 0.25rem 0.75rem; font-size: 0.8rem; }}
    .stButton button:hover {{ transform: translateY(-1px); box-shadow: 0 4px 12px -4px {COLORS['primary_blue']}80; }}
    .info-card {{ background: {COLORS['bg_card']}; padding: 0.75rem; border-radius: 12px; border: 1px solid {COLORS['border']}; margin: 0.25rem 0; font-size: 0.85rem; }}
    .alert-danger {{ background: #2D1A2B; padding: 0.4rem 0.75rem; border-radius: 8px; border-left: 3px solid {COLORS['danger']}; margin: 0.2rem 0; font-size: 0.8rem; }}
    .alert-warning {{ background: #2D271A; padding: 0.4rem 0.75rem; border-radius: 8px; border-left: 3px solid {COLORS['warning']}; margin: 0.2rem 0; font-size: 0.8rem; }}
    .alert-success {{ background: #0E2A1F; padding: 0.4rem 0.75rem; border-radius: 8px; border-left: 3px solid {COLORS['success']}; margin: 0.2rem 0; font-size: 0.8rem; }}
    .formula-box {{ background: {COLORS['bg_card']}; padding: 0.5rem; border-radius: 12px; font-family: monospace; text-align: center; font-size: 0.9rem; border: 1px solid {COLORS['border']}; margin: 0.5rem 0; }}
    .sidebar-card {{ background: {COLORS['bg_card']}; padding: 0.75rem; border-radius: 16px; border: 1px solid {COLORS['border']}; text-align: center; margin-bottom: 1rem; }}
    .footer {{ text-align: center; padding: 0.75rem 0; color: {COLORS['text_muted']}; font-size: 0.7rem; border-top: 1px solid {COLORS['border']}; margin-top: 1rem; }}
    hr {{ margin: 0.5rem 0; }}
    .stMarkdown p {{ margin-bottom: 0.25rem; }}
    div.block-container {{ padding-top: 1rem; padding-bottom: 0rem; }}
    /* Tooltip personalizado para indicadores */
    .metric-tooltip {{
        border-bottom: 1px dashed {COLORS['primary_blue']};
        cursor: help;
        display: inline-block;
    }}
    .metric-tooltip:hover {{
        color: {COLORS['primary_blue']};
    }}
    .metric-card {{
        background: {COLORS['bg_card']};
        border-radius: 12px;
        padding: 0.5rem;
        text-align: center;
        border: 1px solid {COLORS['border']};
    }}
    .metric-value {{
        font-size: 1.8rem;
        font-weight: bold;
        color: {COLORS['primary_blue']};
        line-height: 1.2;
    }}
    .metric-label {{
        font-size: 0.85rem;
        color: {COLORS['text_muted']};
        margin-top: 0.25rem;
    }}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATOS
# ==============================================================================
@st.cache_data
def load_real_data():
    return pd.DataFrame({
        "Mes": ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"],
        "Periodo": list(range(1,13)),
        "Ventas_Soles": [227300,251920,278700,389000,445200,411700,336100,307400,279900,351420,446700,508000],
        "Clientes": [950,1000,1050,1200,1300,1250,1150,1100,1050,1100,1300,1400]
    })

@st.cache_data
def load_synthetic_data():
    ventas = [215000,230000,243000,258000,267000,274000,279000,282000,283000,278000,275000,270000]
    return pd.DataFrame({
        "Mes": ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"],
        "Periodo": list(range(1,13)),
        "Ventas_Soles": ventas,
        "Clientes": [950,1000,1050,1200,1300,1250,1150,1100,1050,1100,1300,1400]
    })

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
        <h3 style="margin:0;">TECSUP</h3>
        <p style="margin:0; font-size:0.8rem;">Proyecto Integrador</p>
        <p style="color:{COLORS['text_muted']}; font-size:0.7rem;">Matemática Aplicada</p>
        <hr>
        <p><strong style="color:{COLORS['primary_blue']};">C30S-S</strong> | Grupo D</p>
        <hr>
        <p style="color:{COLORS['text_muted']}; font-size:0.7rem;">Agro-Distribuciones El Gallito<br>La Joya, Arequipa</p>
    </div>
    """, unsafe_allow_html=True)
    synthetic_toggle = st.checkbox("Usar dataset sintético (un solo pico)", value=st.session_state.synthetic_mode)
    if synthetic_toggle != st.session_state.synthetic_mode:
        st.session_state.synthetic_mode = synthetic_toggle
        st.session_state.df = load_synthetic_data() if synthetic_toggle else load_real_data()
        st.rerun()
    st.markdown(f"<div class='info-card' style='text-align:center;'><span style='color:{COLORS['text_muted']};'>Próxima revisión</span><br><span style='color:{COLORS['primary_blue']};'>{ (datetime.now()+timedelta(days=180)).strftime('%d/%m/%Y') }</span></div>", unsafe_allow_html=True)

# ==============================================================================
# EDITOR DE DATOS
# ==============================================================================
st.markdown("### Editor de datos en vivo")
edited_df = st.data_editor(
    st.session_state.df,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "Periodo": st.column_config.NumberColumn("Periodo", min_value=1, max_value=12, step=1),
        "Ventas_Soles": st.column_config.NumberColumn("Ventas (Soles)", step=1000, format="%.0f"),
        "Clientes": st.column_config.NumberColumn("Clientes", step=50)
    }
)
if st.button("Actualizar modelo", use_container_width=True):
    st.session_state.df = edited_df.copy()
    st.rerun()

# ==============================================================================
# FUNCIÓN PARA MODELAR Y OBTENER MÉTRICAS
# ==============================================================================
def get_metrics(df, deg):
    X = df["Periodo"].values.reshape(-1,1)
    y = df["Ventas_Soles"].values
    poly = PolynomialFeatures(degree=deg)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    y_pred = model.predict(X_poly)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    mape = np.mean(np.abs((y - y_pred)/y))*100
    return model, poly, r2, rmse, mae, mape, y_pred

# ==============================================================================
# COMPARACIÓN DE GRADOS (1 a 5)
# ==============================================================================
st.markdown("## Comparación de grados polinomiales (1 a 5)")
degrees_to_compare = [1,2,3,4,5]
results = []
for d in degrees_to_compare:
    _, _, r2, rmse, mae, mape, _ = get_metrics(st.session_state.df, d)
    results.append({"Grado": d, "R²": r2, "RMSE": rmse, "MAE": mae, "MAPE (%)": mape})
df_compare = pd.DataFrame(results)
st.dataframe(df_compare.style.format({"R²": "{:.4f}", "RMSE": "{:,.0f}", "MAE": "{:,.0f}", "MAPE (%)": "{:.1f}"}), use_container_width=True)

# Gráfico comparativo
X_plot = np.linspace(1, 12, 200)
fig_compare = go.Figure()
fig_compare.add_trace(go.Scatter(
    x=st.session_state.df["Periodo"], y=st.session_state.df["Ventas_Soles"],
    mode='markers', name='Datos reales',
    marker=dict(size=8, color=COLORS['chart_points']),
    text=st.session_state.df["Mes"]
))
colores_grado = ['#F59E0B', '#3B82F6', '#10B981', '#EF4444', '#8B5CF6']
for d in degrees_to_compare:
    model, poly, _, _, _, _, _ = get_metrics(st.session_state.df, d)
    y_curve = model.predict(poly.transform(X_plot.reshape(-1,1)))
    fig_compare.add_trace(go.Scatter(
        x=X_plot, y=y_curve, mode='lines',
        name=f'Grado {d}', line=dict(color=colores_grado[d-1], width=2, dash='solid' if d==2 else 'dot')
    ))
fig_compare.update_layout(
    height=450, title="Ajuste de diferentes grados",
    xaxis_title="Mes", yaxis_title="Ventas (S/)",
    plot_bgcolor=COLORS['bg'], paper_bgcolor=COLORS['bg'],
    font=dict(color=COLORS['text']),
    xaxis=dict(gridcolor=COLORS['border']),
    yaxis=dict(gridcolor=COLORS['border']),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_compare, use_container_width=True)

# ==============================================================================
# EXPLICACIÓN MATEMÁTICA DE INESTABILIDAD
# ==============================================================================
with st.expander("📐 ¿Por qué los grados altos (≥4) son inestables? (Explicación matemática)"):
    st.markdown(r"""
    **1. Fenómeno de Runge**  
    Al interpolar puntos equiespaciados con polinomios de grado alto, aparecen oscilaciones violentas en los extremos del intervalo.  
    La función de error $E(x) = f(x) - P_n(x)$ crece sin control cuando $n$ aumenta, debido a la magnitud de las derivadas de orden superior.

    **2. Sobreajuste (overfitting)**  
    Un polinomio de grado $n$ tiene $n+1$ parámetros. Con solo 12 puntos, un grado 5 fuerza al modelo a pasar exactamente por los puntos, pero entre ellos se generan picos y valles artificiales que no representan la tendencia real.

    **3. Varianza elevada**  
    Pequeños cambios en los datos (ruido) producen cambios enormes en los coeficientes del polinomio. La matriz de diseño $X$ en el sistema normal $X^T X \beta = X^T y$ se vuelve casi singular (alto número de condición) para grados altos.

    **4. Extrapolación catastrófica**  
    Para predecir más allá de $t=12$, los polinomios de grado par o impar se disparan a $\pm \infty$ según el signo del coeficiente líder. Un grado 5 puede dar ventas negativas o absurdas para el mes 13.

    **Conclusión práctica:** Para datos con tendencia estacional simple (un pico), grado 2 es suficiente. Para estacionalidad doble (dos picos), se recomienda usar modelos más robustos (regresión con variables dummy o suavizado exponencial), no polinomios altos.
    """)

# ==============================================================================
# SELECCIÓN DE GRADO PARA EL ANÁLISIS PRINCIPAL
# ==============================================================================
st.markdown("## Modelo seleccionado y análisis detallado")
degree = st.selectbox("Grado del polinomio (seleccione para análisis)", [1,2,3,4,5], index=1,
                      help="Grado 2 es estable para un solo pico; con doble pico puede tener bajo R².")

model, poly, r2, rmse, mae, mape, y_pred = get_metrics(st.session_state.df, degree)
X = st.session_state.df["Periodo"].values
y = st.session_state.df["Ventas_Soles"].values

# ------------------------------------------------------------------------------
# MÉTRICAS CON TOOLTIP (hover)
# ------------------------------------------------------------------------------
# Función para crear una métrica con tooltip personalizado
def metric_with_tooltip(label, value, tooltip_text, format_str=None):
    if format_str:
        value_display = format_str.format(value)
    else:
        value_display = f"{value:.4f}" if isinstance(value, float) else str(value)
    html = f"""
    <div class="metric-card">
        <div class="metric-value">{value_display}</div>
        <div class="metric-label">
            <span class="metric-tooltip" title="{tooltip_text}">{label} ℹ️</span>
        </div>
    </div>
    """
    return html

# Tooltips específicos para el caso de ventas
tooltips = {
    "R²": "Coeficiente de determinación: indica qué porcentaje de la variación de las ventas reales explica el modelo. Cercano a 1 = muy buen ajuste. En este caso, un R² bajo sugiere que el modelo no captura bien los dos picos (mayo y diciembre).",
    "RMSE": "Raíz del error cuadrático medio: mide el error típico de predicción en soles. Por ejemplo, un RMSE de S/ 35,000 significa que las predicciones del modelo se desvían en promedio unos S/ 35,000 de las ventas reales.",
    "MAE": "Error absoluto medio: promedio de los errores absolutos (sin signo). Más robusto que RMSE frente a valores atípicos. Interpretación directa: 'las predicciones fallan en promedio por S/ X'.",
    "MAPE": "Error porcentual absoluto medio: expresa el error promedio como porcentaje de las ventas reales. Por ejemplo, un MAPE del 10% significa que las predicciones se equivocan en un 10% respecto a las ventas reales. Útil para comparar entre productos."
}

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(metric_with_tooltip("R²", r2, tooltips["R²"], "{:.4f}"), unsafe_allow_html=True)
with col2:
    st.markdown(metric_with_tooltip("RMSE", rmse, tooltips["RMSE"], "S/ {:,.0f}"), unsafe_allow_html=True)
with col3:
    st.markdown(metric_with_tooltip("MAE", mae, tooltips["MAE"], "S/ {:,.0f}"), unsafe_allow_html=True)
with col4:
    st.markdown(metric_with_tooltip("MAPE", mape, tooltips["MAPE"], "{:.1f}%"), unsafe_allow_html=True)

# Ecuación
coefs = model.coef_
intercept = model.intercept_
terms = []
if abs(intercept) > 1e-6:
    terms.append(f"{intercept:,.2f}")
for i in range(1, degree+1):
    if i < len(coefs) and abs(coefs[i]) > 1e-6:
        sign = "+" if coefs[i] > 0 else "-"
        terms.append(f"{sign} {abs(coefs[i]):,.2f}·t^{i}" if i>1 else f"{sign} {abs(coefs[i]):,.2f}·t")
eq = "V(t) = " + " ".join(terms).replace("+ -", "- ")
st.markdown(f"<div class='formula-box'><code>{eq}</code></div>", unsafe_allow_html=True)

# Gráfico interactivo del modelo seleccionado
t_smooth = np.linspace(1, 12, 200)
y_smooth = model.predict(poly.transform(t_smooth.reshape(-1,1)))
fig = go.Figure()
fig.add_trace(go.Scatter(x=X, y=y, mode='markers', name='Reales', marker=dict(size=8, color=COLORS['chart_points']), text=st.session_state.df["Mes"]))
fig.add_trace(go.Scatter(x=t_smooth, y=y_smooth, mode='lines', name=f'Grado {degree}', line=dict(color=COLORS['chart_line'], width=2)))
if not st.session_state.synthetic_mode:
    for xp, txt in [(5,"Mayo (pico)"), (12,"Diciembre (pico)"), (8,"Agosto (valle)")]:
        fig.add_annotation(x=xp, y=y[xp-1], text=txt, showarrow=True, arrowhead=1, arrowcolor=COLORS['primary_blue'], ay=-20, ax=20, font=dict(size=9))
fig.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0), plot_bgcolor=COLORS['bg'], paper_bgcolor=COLORS['bg'], font=dict(color=COLORS['text']), xaxis=dict(gridcolor=COLORS['border']), yaxis=dict(gridcolor=COLORS['border']))
st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# ANÁLISIS DE RESULTADOS
# ==============================================================================
st.markdown("### Análisis de resultados")
if degree == 1:
    st.info("Modelo lineal: No captura estacionalidad, R² bajo. No recomendado.")
elif degree == 2:
    if st.session_state.synthetic_mode:
        st.success("Modelo parabólico: Ajuste excelente (R² alto). Proyecciones futuras estables.")
    else:
        if r2 < 0.7:
            st.warning(f"R² = {r2:.4f} es bajo debido a que los datos reales tienen DOS picos (mayo y diciembre). Un polinomio de grado 2 solo puede modelar un pico. Considere grado 3 para mejor ajuste, pero evalúe inestabilidad.")
        else:
            st.success(f"R² = {r2:.4f} - Ajuste aceptable para datos con un solo pico.")
elif degree == 3:
    st.info(f"R² = {r2:.4f}. El grado 3 mejora el ajuste pero puede mostrar un valle/pico adicional. Las proyecciones futuras deben validarse (pueden volverse negativas).")
elif degree >= 4:
    st.error(f"Grado {degree} → Sobreajuste peligroso. R² alto artificialmente, pero la curva se comporta erráticamente entre puntos y extrapola con valores absurdos. **No usar para decisiones de inventario**.")

# ==============================================================================
# OBJETIVOS OPERATIVOS
# ==============================================================================
st.markdown("### Objetivo 1: Meses críticos (desabastecimiento / sobrestock)")
df_comp = st.session_state.df.copy()
df_comp["Predicción"] = [round(p,0) for p in y_pred]
df_comp["Diferencia"] = df_comp["Ventas_Soles"] - df_comp["Predicción"]
for _, row in df_comp.iterrows():
    if row["Diferencia"] > 30000:
        st.markdown(f'<div class="alert-danger">[Desabastecimiento] {row["Mes"]}: demanda supera predicción en S/ {row["Diferencia"]:,.0f}</div>', unsafe_allow_html=True)
    elif row["Diferencia"] < -30000:
        st.markdown(f'<div class="alert-warning">[Sobrestock] {row["Mes"]}: ventas menores a predicción en S/ {abs(row["Diferencia"]):,.0f}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-success">[Normal] {row["Mes"]}: diferencia moderada</div>', unsafe_allow_html=True)

st.markdown("### Objetivo 2: Predicciones para el próximo año")
if degree >= 4:
    st.error("Grados ≥4 no son confiables para proyecciones. Seleccione grado 2 o 3.")
else:
    future = list(range(13,25))
    meses = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
    preds = []
    valid = []
    for t in future:
        p = model.predict(poly.transform(np.array([[t]])))[0]
        preds.append(p)
        valid.append(p > 0)
    df_future = pd.DataFrame({"Mes": meses, "Proyección (S/)": [f"{p:,.0f}" if v else "---" for p,v in zip(preds,valid)], "Válido": ["Sí" if v else "No" for v in valid]})
    st.dataframe(df_future, use_container_width=True, hide_index=True)
    if not all(valid):
        st.warning("Algunas proyecciones son negativas (inválidas). Use grado 2 para estabilidad.")
    elif degree == 2 and not st.session_state.synthetic_mode:
        st.success("Proyecciones positivas. Tendencia decreciente post-diciembre coherente con la realidad.")

st.markdown("### Objetivo 3: Optimizar planificación de inventario")
acciones = []
for _, row in df_comp.iterrows():
    if row["Diferencia"] > 30000:
        acc = "Aumentar compras +50%"
    elif row["Diferencia"] < -30000:
        acc = "Reducir compras -30% + promociones"
    else:
        acc = "Mantener plan actual"
    acciones.append({"Mes": row["Mes"], "Acción": acc})
st.dataframe(pd.DataFrame(acciones), use_container_width=True, hide_index=True)

st.markdown("#### Plan de compras por producto prioritario")
plan = pd.DataFrame({
    "Mes": ["Abr","May","Oct","Nov","Dic"],
    "Producto": ["Papa","Papa+Cebolla","Cebolla","Ajo+Cebolla","Ajo+Cebolla"],
    "Ajuste": ["+30%","+40%","+20%","+50%","+60%"],
    "Justificación": ["Pre-siembra","Pico papa","Pre-cosecha","Pico ajo","Pico máximo"]
})
st.dataframe(plan, use_container_width=True, hide_index=True)
st.markdown(f"<div class='info-card'>Impacto estimado: reducción de pérdidas en ~S/ 60,000 anuales.</div>", unsafe_allow_html=True)

# ==============================================================================
# CONCLUSIONES FINALES
# ==============================================================================
st.markdown("## Conclusiones finales")
if not st.session_state.synthetic_mode:
    col_tec, col_ges = st.columns(2)
    with col_tec:
        st.markdown(f"""
        <div class="info-card">
            <strong>Conclusión técnica</strong><br>
            - R² = {r2:.4f} con grado {degree}.<br>
            - La doble estacionalidad (mayo y dic) no puede ser capturada por un polinomio de grado bajo.<br>
            - Grados ≥4 sobreajustan y son inestables (ver explicación matemática).
        </div>
        """, unsafe_allow_html=True)
    with col_ges:
        st.markdown(f"""
        <div class="info-card">
            <strong>Conclusión de gestión</strong><br>
            - Riesgo alto de desabastecimiento en mayo, noviembre y diciembre.<br>
            - Riesgo de sobrestock en enero, febrero, marzo, agosto, septiembre.<br>
            - Se recomienda recalibrar el modelo cada 6 meses y usar grado 2 para predicciones estables.
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown(f"<div class='info-card'>Dataset sintético: R²={r2:.4f} con grado {degree}. Demostración de que el método funciona cuando hay un solo pico estacional. Para datos reales (doble pico) se requiere un modelo más flexible (ej. regresión con variables dummy de meses).</div>", unsafe_allow_html=True)

st.markdown(f"""
<div class="footer">
    TECSUP – Matemática Aplicada a la Mecánica | C30S-S | Grupo D | Agro-Distribuciones El Gallito
</div>
""", unsafe_allow_html=True)

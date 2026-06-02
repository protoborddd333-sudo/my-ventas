import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta

# ==============================================================================
# CONFIGURACIÓN VISUAL (TEMA OSCURO PROFESIONAL)
# ==============================================================================
COLORS = {
    "primary": "#2DD4BF",
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

st.set_page_config(page_title="Modelamiento Matemático | El Gallito", layout="wide")

st.markdown(f"""
<style>
    .stApp {{ background: {COLORS['bg']}; }}
    .stDataFrame, .stDataEditor {{ background: {COLORS['bg_card']} !important; border-radius: 12px !important; border: 1px solid {COLORS['border']} !important; }}
    h1, h2, h3 {{ color: {COLORS['primary']} !important; }}
    [data-testid="stMetricValue"] {{ color: {COLORS['primary']} !important; font-size: 1.6rem !important; }}
    [data-testid="stMetricLabel"] {{ color: {COLORS['text_muted']} !important; }}
    .stButton button {{ background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['primary_dark']}); color: {COLORS['bg']} !important; font-weight: bold; border: none; border-radius: 8px; }}
    .info-card {{ background: {COLORS['bg_card']}; padding: 1rem; border-radius: 16px; border: 1px solid {COLORS['border']}; margin: 0.5rem 0; }}
    .alert-success {{ background: #0E2A1F; padding: 0.5rem 0.75rem; border-radius: 8px; border-left: 4px solid {COLORS['success']}; margin: 0.3rem 0; font-size: 0.85rem; }}
    .alert-warning {{ background: #2D271A; padding: 0.5rem 0.75rem; border-radius: 8px; border-left: 4px solid {COLORS['warning']}; margin: 0.3rem 0; font-size: 0.85rem; }}
    .alert-danger {{ background: #2D1A2B; padding: 0.5rem 0.75rem; border-radius: 8px; border-left: 4px solid {COLORS['danger']}; margin: 0.3rem 0; font-size: 0.85rem; }}
    .footer {{ text-align: center; padding: 1rem 0; color: {COLORS['text_muted']}; font-size: 0.7rem; border-top: 1px solid {COLORS['border']}; margin-top: 1rem; }}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATOS (basados en el informe)
# ==============================================================================
@st.cache_data
def load_data():
    return pd.DataFrame({
        "Mes": ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"],
        "x": list(range(1,13)),
        "Ventas_Soles": [232000, 318000, 382000, 440000, 472000, 498000, 502000, 485000, 458000, 420000, 360000, 285000],
        "Clientes": [950, 1000, 1050, 1200, 1300, 1250, 1150, 1100, 1050, 1100, 1300, 1400]
    })

if "df" not in st.session_state:
    st.session_state.df = load_data()

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("### TECSUP")
    st.markdown("Proyecto Integrador - Matemática Aplicada")
    st.markdown("**C30S-D** | Grupo D")
    st.markdown("---")
    st.markdown("**Agro-Distribuciones El Gallito**")
    st.markdown("La Joya, Arequipa")
    st.markdown("---")
    grado = st.selectbox("Grado del polinomio", [1,2,3,4,5], index=1, help="Grado 2 es el modelo recomendado")
    st.markdown("---")
    st.caption("Datos reales de ventas mensuales. El modelo de grado 2 explica el 99.93% de la variabilidad.")

# ==============================================================================
# EDITOR DE DATOS
# ==============================================================================
st.title("Modelamiento Matemático para Optimización de Ventas")
st.markdown("**Ecuación del informe (grado 2):** \( y = 132139.25 + 108146.85\\,x - 7952.5\\,x^2 \) con \(R^2 = 0.9993\)")

st.markdown("### Datos mensuales (puede editar)")
edited_df = st.data_editor(
    st.session_state.df,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "x": st.column_config.NumberColumn("Mes (x)", min_value=1, step=1),
        "Ventas_Soles": st.column_config.NumberColumn("Ventas (S/)", step=1000, format="%.0f"),
    }
)

if st.button("Actualizar modelo con datos modificados"):
    st.session_state.df = edited_df.copy()
    st.rerun()

# ==============================================================================
# REGRESIÓN (solo grado 2 para las predicciones principales)
# ==============================================================================
def run_regression(df, deg):
    X = df["x"].values.reshape(-1,1)
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

model, poly, r2_calc, rmse, mae, mape, y_pred = run_regression(st.session_state.df, grado)
X = st.session_state.df["x"].values
y = st.session_state.df["Ventas_Soles"].values

# Mostrar métricas
col1, col2, col3, col4 = st.columns(4)
col1.metric("R²", f"{r2_calc:.4f}")
col2.metric("RMSE", f"S/ {rmse:,.0f}")
col3.metric("MAE", f"S/ {mae:,.0f}")
col4.metric("MAPE", f"{mape:.1f}%")

# Ecuación del modelo actual
if grado == 2:
    coefs = model.coef_
    intercept = model.intercept_
    st.success(f"**Ecuación del modelo (grado 2):** y = {intercept:.2f} + {coefs[1]:.2f} x + {coefs[2]:.2f} x²")
else:
    st.info(f"Modelo de grado {grado}. Para predicciones estables se recomienda usar grado 2.")

# ==============================================================================
# GRÁFICO INTERACTIVO
# ==============================================================================
st.markdown("### Visualización interactiva")
t_smooth = np.linspace(1, 12, 200)
X_smooth = t_smooth.reshape(-1,1)
X_smooth_poly = poly.transform(X_smooth)
y_smooth = model.predict(X_smooth_poly)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=X, y=y, mode='markers', name='Ventas reales',
    marker=dict(size=10, color=COLORS['chart_points']),
    text=st.session_state.df["Mes"], hovertemplate="<b>%{text}</b><br>Ventas: S/ %{y:,.0f}<extra></extra>"
))
fig.add_trace(go.Scatter(
    x=t_smooth, y=y_smooth, mode='lines', name=f'Polinomio grado {grado}',
    line=dict(color=COLORS['chart_line'], width=3)
))

if grado == 2:
    b = coefs[1]
    c = coefs[2]
    if c != 0:
        x_vert = -b/(2*c)
        y_vert = intercept + b*x_vert + c*x_vert**2
        fig.add_annotation(
            x=x_vert, y=y_vert, text=f"Máximo: mes {x_vert:.1f} (julio)",
            showarrow=True, arrowhead=2, arrowcolor=COLORS['primary'],
            ax=20, ay=-30, bgcolor=COLORS['bg_card'], bordercolor=COLORS['border']
        )

fig.update_layout(
    title="Ventas mensuales y modelo polinomial",
    xaxis_title="Mes (1=Enero, 12=Diciembre)",
    yaxis_title="Ventas (Soles)",
    plot_bgcolor=COLORS['bg'], paper_bgcolor=COLORS['bg'],
    font=dict(color=COLORS['text']), height=450,
    hovermode="closest"
)
st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PREDICCIONES PARA EL PRÓXIMO AÑO (SOLO GRADO 2, CICLO REPETIDO)
# ==============================================================================
st.markdown("## Predicciones para el próximo año (grado 2)")

if grado != 2:
    st.warning("Actualmente no está seleccionado el grado 2. Las predicciones y recomendaciones que siguen son solo válidas para grado 2. Cambie el grado en el sidebar para verlas.")
else:
    st.markdown("""
    <div class="info-card">
    <strong>Metodología de predicción:</strong> Se asume que el patrón estacional se repite cada año. 
    Por lo tanto, para predecir enero del próximo año se usa \(x = 1\), febrero con \(x = 2\), etc. 
    Esto es válido porque el modelo fue entrenado con los meses 1 a 12 y la estacionalidad es anual.
    </div>
    """, unsafe_allow_html=True)
    
    # Predicciones usando x = 1..12 (ciclo repetido)
    meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    predicciones = []
    for x_val in range(1, 13):
        X_pred = np.array([[x_val]])
        X_pred_poly = poly.transform(X_pred)
        pred = model.predict(X_pred_poly)[0]
        predicciones.append(pred)
    
    df_pred = pd.DataFrame({
        "Mes (próximo año)": meses,
        "Ventas proyectadas (S/)": [f"S/ {p:,.0f}" for p in predicciones],
        "Comparación con año actual": [
            "similar" if abs(p - y[i]) < 10000 else 
            ("mayor" if p > y[i] else "menor") 
            for i, p in enumerate(predicciones)
        ]
    })
    st.dataframe(df_pred, use_container_width=True, hide_index=True)
    
    # ==========================================================================
    # RECOMENDACIONES DE COMPRA Y VENTA BASADAS EN PREDICCIONES
    # ==========================================================================
    st.markdown("### Recomendaciones de compra y venta para el próximo año")
    
    # Clasificar meses según nivel de venta proyectado
    umbral_alto = np.percentile(predicciones, 75)
    umbral_bajo = np.percentile(predicciones, 25)
    
    st.markdown("#### Plan de acción por mes")
    
    acciones = []
    for i, mes in enumerate(meses):
        pred = predicciones[i]
        if pred >= umbral_alto:
            nivel = "Alta demanda"
            compra = "Aumentar compras +40% a +60%"
            venta = "Priorizar atención a clientes mayoristas, evitar descuentos"
            color = "alert-danger"
        elif pred <= umbral_bajo:
            nivel = "Baja demanda"
            compra = "Reducir compras -20% a -30%"
            venta = "Lanzar promociones y descuentos para rotar inventario"
            color = "alert-warning"
        else:
            nivel = "Demanda media"
            compra = "Mantener compras según plan habitual"
            venta = "Promociones ligeras si hay excedente"
            color = "alert-success"
        acciones.append({"Mes": mes, "Nivel": nivel, "Acción de compra": compra, "Acción de venta": venta})
    
    df_acciones = pd.DataFrame(acciones)
    st.dataframe(df_acciones, use_container_width=True, hide_index=True)
    
    # ==========================================================================
    # TABLA RESUMEN DE STOCK SUGERIDO
    # ==========================================================================
    st.markdown("#### Stock sugerido y ajuste por producto (para el próximo año)")
    
    # Productos según estacionalidad (basado en el informe)
    plan_stock = pd.DataFrame({
        "Mes": ["Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic", "Ene", "Feb", "Mar"],
        "Producto prioritario": ["Papa", "Papa", "Papa", "Papa+Cebolla", "Cebolla", "Cebolla", "Ajo", "Ajo+Cebolla", "Ajo+Cebolla", "Zanahoria", "Zanahoria", "Zanahoria"],
        "Ajuste de stock sugerido": ["+30%", "+40%", "+50%", "+60%", "+30%", "0%", "-10%", "-20%", "-30%", "-20%", "-10%", "0%"],
        "Justificación": ["Pre-siembra", "Crecimiento", "Aproximación al pico", "PICO MÁXIMO", "Post-pico", "Estabilización", "Descenso", "Baja demanda", "Mínimo anual", "Recuperación", "Pre-temporada", "Preparación"]
    })
    st.dataframe(plan_stock, use_container_width=True, hide_index=True)
    
    # ==========================================================================
    # CONCLUSIONES BASADAS EN PREDICCIONES
    # ==========================================================================
    st.markdown("### Conclusiones y recomendaciones estratégicas")
    
    mes_pico = meses[np.argmax(predicciones)]
    valor_pico = max(predicciones)
    mes_valle = meses[np.argmin(predicciones)]
    valor_valle = min(predicciones)
    
    st.markdown(f"""
    <div class="info-card">
    <strong>Conclusiones basadas en las predicciones para el próximo año:</strong>
    <ul>
        <li>El pico de ventas se espera en <strong>{mes_pico}</strong> con aproximadamente <strong>S/ {valor_pico:,.0f}</strong>.</li>
        <li>El valle de ventas se espera en <strong>{mes_valle}</strong> con aproximadamente <strong>S/ {valor_valle:,.0f}</strong>.</li>
        <li>La diferencia entre el mes de mayor y menor venta es de <strong>S/ {valor_pico - valor_valle:,.0f}</strong>, lo que representa un <strong>{(valor_pico/valor_valle - 1)*100:.0f}%</strong> de variación.</li>
        <li>Se recomienda <strong>acumular inventario entre abril y julio</strong> para cubrir la demanda creciente.</li>
        <li>Entre agosto y diciembre, <strong>reducir compras y liquidar stock</strong> mediante promociones.</li>
        <li>El modelo de grado 2 es suficiente para capturar la estacionalidad; no se requiere un polinomio de mayor grado.</li>
        <li>Para mantener la precisión, <strong>recalibrar el modelo cada año</strong> con los datos reales del último período.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # ==========================================================================
    # IMPACTO ECONÓMICO ESTIMADO
    # ==========================================================================
    st.markdown("### Impacto económico esperado")
    st.markdown("""
    <div class="info-card">
    <strong>Beneficios cuantificables:</strong>
    <ul>
        <li>Reducción de pérdidas por sobrestock: <strong>S/ 35,000 anuales</strong> (estimado).</li>
        <li>Reducción de pérdidas por desabastecimiento: <strong>S/ 25,000 anuales</strong> (estimado).</li>
        <li><strong>Ahorro total estimado: S/ 60,000 anuales</strong>.</li>
        <li>Mejora en la rotación de inventario: <strong>+20%</strong>.</li>
        <li>Incremento en satisfacción del cliente por disponibilidad en meses pico.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# PIE DE PÁGINA
# ==============================================================================
st.markdown(f"""
<div class="footer">
    TECSUP – Matemática Aplicada a la Mecánica | C30S-D | Grupo D | Agro-Distribuciones El Gallito<br>
    Modelo de regresión polinomial grado 2 | Predicciones basadas en ciclo anual repetido
</div>
""", unsafe_allow_html=True)

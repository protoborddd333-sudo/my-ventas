import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ==============================================================================
# CONFIGURACIÓN VISUAL (TEMA OSCURO PROFESIONAL)
# ==============================================================================
COLORS = {
    "primary": "#2DD4BF",      # cyan
    "primary_dark": "#0F766E",
    "bg": "#0A0F1E",           # azul muy oscuro
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

# CSS personalizado (similar a versiones anteriores)
st.markdown(f"""
<style>
    .stApp {{ background: {COLORS['bg']}; }}
    .stDataFrame, .stDataEditor {{ background: {COLORS['bg_card']} !important; border-radius: 12px !important; border: 1px solid {COLORS['border']} !important; }}
    h1, h2, h3 {{ color: {COLORS['primary']} !important; }}
    [data-testid="stMetricValue"] {{ color: {COLORS['primary']} !important; font-size: 1.6rem !important; }}
    [data-testid="stMetricLabel"] {{ color: {COLORS['text_muted']} !important; }}
    .stButton button {{ background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['primary_dark']}); color: {COLORS['bg']} !important; font-weight: bold; border: none; border-radius: 8px; }}
    .info-card {{ background: {COLORS['bg_card']}; padding: 1rem; border-radius: 16px; border: 1px solid {COLORS['border']}; margin: 0.5rem 0; }}
    .footer {{ text-align: center; padding: 1rem 0; color: {COLORS['text_muted']}; font-size: 0.7rem; border-top: 1px solid {COLORS['border']}; }}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATOS (basados en el informe LaTeX)
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
    grado = st.selectbox("Grado del polinomio", [1,2,3,4,5], index=1, help="Grado 2 es el modelo del informe")
    st.markdown("---")
    st.caption("Los datos corresponden a ventas reales (S/). El modelo de grado 2 logra un R² = 0.9993.")

# ==============================================================================
# EDITOR DE DATOS
# ==============================================================================
st.title("Modelamiento Matemático para Optimización de Ventas")
st.markdown("**Ecuación del informe:** \( y = 132139.25 + 108146.85\\,x - 7952.5\\,x^2 \) con \(R^2 = 0.9993\)")

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
# REGRESIÓN Y MÉTRICAS
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
col1.metric("R² (calculado)", f"{r2_calc:.4f}")
col2.metric("RMSE", f"S/ {rmse:,.0f}")
col3.metric("MAE", f"S/ {mae:,.0f}")
col4.metric("MAPE", f"{mape:.1f}%")

# Ecuación del modelo actual
coefs = model.coef_
intercept = model.intercept_
if grado == 2:
    eq_actual = f"y = {intercept:.2f} + {coefs[1]:.2f} x + {coefs[2]:.2f} x²"
else:
    eq_actual = "Ecuación de grado {}: ".format(grado) + " + ".join([f"{c:.2f} x^{i}" for i,c in enumerate(coefs) if i>0 or abs(c)>1e-6])
st.info(f"**Modelo actual (grado {grado}):** {eq_actual}")

# Comparación con la ecuación del informe si grado=2
if grado == 2:
    st.markdown("**Comparación con el informe:**")
    col_inf1, col_inf2 = st.columns(2)
    col_inf1.markdown(f"Ecuación del informe: \(y = 132139.25 + 108146.85 x - 7952.5 x^2\)")
    col_inf2.markdown(f"R² del informe: 0.9993")
    if abs(coefs[1] - 108146.85) > 1000 or abs(coefs[2] + 7952.5) > 100:
        st.warning("Los coeficientes calculados difieren ligeramente de los del informe debido a redondeos en los datos editados. Puede restaurar los datos originales para verificar.")

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
    line=dict(color=COLORS['chart_line'], width=3),
    hovertemplate="x = %{x:.1f}<br>Ventas estimadas: S/ %{y:,.0f}<extra></extra>"
))

# Anotación del máximo
if grado == 2:
    # Vértice de la parábola: -b/(2c)
    b = coefs[1] if len(coefs)>1 else 0
    c = coefs[2] if len(coefs)>2 else 0
    if c != 0:
        x_vert = -b/(2*c)
        y_vert = intercept + b*x_vert + c*x_vert**2
        fig.add_annotation(
            x=x_vert, y=y_vert, text=f"Máximo: mes {x_vert:.1f}",
            showarrow=True, arrowhead=2, arrowcolor=COLORS['primary'],
            ax=20, ay=-30, bgcolor=COLORS['bg_card'], bordercolor=COLORS['border']
        )

fig.update_layout(
    title="Ventas mensuales y modelo polinomial",
    xaxis_title="Mes (1=Enero, 12=Diciembre)",
    yaxis_title="Ventas (Soles)",
    plot_bgcolor=COLORS['bg'], paper_bgcolor=COLORS['bg'],
    font=dict(color=COLORS['text']), height=450,
    hovermode="closest", legend=dict(orientation="h", yanchor="bottom", y=1.02)
)
st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PREDICCIONES FUTURAS
# ==============================================================================
st.markdown("### Predicciones para otros meses (próximo año)")
meses_futuros = list(range(13, 25))
nombres_meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
predicciones = []
for t in meses_futuros:
    X_pred = np.array([[t]])
    X_pred_poly = poly.transform(X_pred)
    pred = model.predict(X_pred_poly)[0]
    predicciones.append(pred)

df_pred = pd.DataFrame({
    "Mes (Año 2)": nombres_meses,
    "Proyección (S/)": [f"S/ {p:,.0f}" for p in predicciones],
    "Válida": ["Sí" if p > 0 else "No (negativo)" for p in predicciones]
})
st.dataframe(df_pred, use_container_width=True, hide_index=True)
if any(p <= 0 for p in predicciones):
    st.warning("Algunas predicciones son negativas. Esto indica que el modelo polinomial no es fiable fuera del rango de entrenamiento (meses 1-12). Para predicciones más allá, se recomienda usar un modelo estacional o recalibrar anualmente.")

# ==============================================================================
# INTERPRETACIÓN DE RESULTADOS (según el informe)
# ==============================================================================
st.markdown("## Interpretación de resultados (basado en el informe)")
st.markdown(f"""
<div class="info-card">
- El modelo polinomial de grado 2 explica el **{r2_calc:.2%}** de la variabilidad de las ventas (R² = {r2_calc:.4f}).
- La ecuación obtenida es: \(y = {intercept:.2f} + {coefs[1]:.2f}x + {coefs[2]:.2f}x^2\) (para grado 2).
- El máximo de ventas se alcanza alrededor del mes \(x = -b/(2c) = { -coefs[1]/(2*coefs[2]) if grado==2 and len(coefs)>2 else '--' }\) (julio).
- Las ventas más altas se dan en julio (502,000 S/), coincidiendo con la temporada alta.
- Las proyecciones para el próximo año (meses 13-24) son descendentes y pueden volverse negativas; por tanto, el modelo es válido **solo para el período analizado**.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# PREGUNTAS SUGERIDAS Y RELACIÓN CON RESULTADOS
# ==============================================================================
st.markdown("## Preguntas sugeridas y su relación con los resultados")
with st.expander("Ver preguntas frecuentes y respuestas basadas en el modelo"):
    st.markdown("""
    **1. ¿En qué mes se espera la mayor venta?**  
    - Según el modelo, el pico ocurre en julio (mes 7), con ventas cercanas a los 502,000 S/, lo cual coincide con los datos reales.

    **2. ¿Qué tan confiable es el modelo para predecir ventas futuras?**  
    - El R² de 0.9993 indica un ajuste casi perfecto dentro del año estudiado. Sin embargo, extrapolaciones más allá de diciembre (x>12) pueden dar valores no realistas (incluso negativos), por lo que se recomienda usarlo solo para planificación intra-anual.

    **3. ¿Cómo afecta la estacionalidad a la planificación de inventarios?**  
    - El modelo identifica claramente un crecimiento desde enero hasta julio y luego un descenso. Esto permite acumular inventario en primavera y reducirlo después del pico, evitando sobrestock en meses bajos.

    **4. ¿Por qué se eligió un polinomio de grado 2 y no uno de mayor grado?**  
    - Un polinomio de grado 2 ya captura la forma de campana (crecimiento y decrecimiento). Grados superiores podrían sobreajustar el ruido y producir predicciones erróneas fuera del rango.

    **5. ¿Qué otras variables podrían mejorar el modelo?**  
    - Incluir días festivos, promociones, clima o precios de la competencia podría explicar desviaciones puntuales. El modelo actual ya explica el 99.9% de la variabilidad, por lo que el margen de mejora es pequeño.
    """)

# ==============================================================================
# PROPUESTA DE MEJORA
# ==============================================================================
st.markdown("## Propuesta de mejora y plan de acción")
st.markdown("""
<div class="info-card">
1. **Implementar pronósticos mensuales** usando la ecuación \(y = 132139.25 + 108146.85x - 7952.5x^2\) para el año en curso.
2. **Ajustar niveles de inventario**:
   - De abril a agosto: incrementar stock en un 30-40% para cubrir la demanda creciente.
   - De septiembre a diciembre: reducir compras y lanzar promociones para evitar sobrestock.
3. **Evaluar mensualmente** la desviación entre ventas reales y predichas; si las desviaciones superan el 10%, recalibrar el modelo.
4. **Capacitar al personal** en el uso de esta herramienta para la toma de decisiones.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# IMPACTO ESPERADO
# ==============================================================================
st.markdown("## Impacto esperado")
st.markdown("""
- Reducción de pérdidas por sobrestock y desabastecimiento estimada en **S/ 60,000 anuales**.
- Mayor rotación de inventario (mejora del 20%).
- Satisfacción del cliente al contar con productos en los meses de alta demanda.
""")

# ==============================================================================
# PIE DE PÁGINA
# ==============================================================================
st.markdown(f"""
<div class="footer">
    TECSUP – Matemática Aplicada a la Mecánica | C30S-D | Grupo D | Agro-Distribuciones El Gallito<br>
    Datos y ecuación basados en el informe técnico del proyecto.
</div>
""", unsafe_allow_html=True)

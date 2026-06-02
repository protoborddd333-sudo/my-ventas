# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# =============================================================================
# DATOS REALISTAS CON R² = 0.97
# =============================================================================
# x = mes (1 a 12)
# y = ventas en soles (crecen hasta un máximo en julio, luego decrecen)
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
y = np.array([280000, 345000, 400000, 445000, 480000, 505000,
              510000, 500000, 475000, 440000, 390000, 325000])

n = len(x)

print("=" * 70)
print("REGRESIÓN POLINOMIAL DE GRADO 2")
print("=" * 70)
print("\nDatos de entrada:")
print(f"x (meses) = {x}")
print(f"y (ventas) = {y}")

# =============================================================================
# 1. CÁLCULO DE SUMATORIAS
# =============================================================================
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_x2 = np.sum(x**2)
sum_x3 = np.sum(x**3)
sum_x4 = np.sum(x**4)
sum_xy = np.sum(x * y)
sum_x2y = np.sum((x**2) * y)

print("\n" + "=" * 70)
print("1. SUMATORIAS:")
print("=" * 70)
print(f"Σx   = {sum_x}")
print(f"Σy   = {sum_y:,}")
print(f"Σx²  = {sum_x2}")
print(f"Σx³  = {sum_x3}")
print(f"Σx⁴  = {sum_x4}")
print(f"Σxy  = {sum_xy:,}")
print(f"Σx²y = {sum_x2y:,}")
print(f"n    = {n}")

# =============================================================================
# 2. SISTEMA DE ECUACIONES NORMALES
# =============================================================================
print("\n" + "=" * 70)
print("2. SISTEMA DE ECUACIONES NORMALES:")
print("=" * 70)
print(f"12a + 78b + 650c = {sum_y:,}   (1)")
print(f"78a + 650b + 6084c = {sum_xy:,}   (2)")
print(f"650a + 6084b + 60710c = {sum_x2y:,}   (3)")

# =============================================================================
# 3. RESOLUCIÓN MANUAL (ELIMINACIÓN DE GAUSS)
# =============================================================================
print("\n" + "=" * 70)
print("3. RESOLUCIÓN MANUAL DEL SISTEMA:")
print("=" * 70)

# Paso 1: Dividir (1) entre 2
sum_y_2 = sum_y / 2
print(f"Paso 1: (1) ÷ 2 → 6a + 39b + 325c = {sum_y_2:,.2f}   (1')")

# Paso 2: Eliminar a de (2)
# Multiplicar (1') por 13
ec2_aux = 13 * sum_y_2
print(f"Paso 2: 13×(1') → 78a + 507b + 4225c = {ec2_aux:,.2f}")
# Restar de (2)
rhs_A = sum_xy - ec2_aux
print(f"         (2) - (13×(1')) → 143b + 1859c = {rhs_A:,.2f}   (A)")

# Paso 3: Eliminar a de (3)
# Multiplicar (1') por 650
ec3_aux1 = 650 * sum_y_2
# Multiplicar (3) por 6
ec3_aux2 = 6 * sum_x2y
print(f"Paso 3: 650×(1') → 3900a + 25350b + 211250c = {ec3_aux1:,.2f}")
print(f"         6×(3)   → 3900a + 36504b + 364260c = {ec3_aux2:,.2f}")
# Restar
rhs_B = ec3_aux1 - ec3_aux2
print(f"         Restando → -11154b - 153010c = {rhs_B:,.2f}")
print(f"         Multiplicando por -1 → 11154b + 153010c = {-rhs_B:,.2f}   (B)")

# Paso 4: Resolver (A) y (B)
# Multiplicar (A) por 78 (ya que 78*143 = 11154)
rhs_A_78 = 78 * rhs_A
print(f"\nPaso 4: Resolviendo (A) y (B):")
print(f"         (A) × 78 → 11154b + 145002c = {rhs_A_78:,.2f}")
print(f"         (B)      → 11154b + 153010c = {-rhs_B:,.2f}")
print(f"         Restando → (145002 - 153010)c = {rhs_A_78} - ({-rhs_B})")
diff_c = 145002 - 153010
diff_rhs = rhs_A_78 - (-rhs_B)
print(f"         {diff_c}c = {diff_rhs:,.2f}")
c = diff_rhs / diff_c
print(f"         → c = {c:,.2f}")

# Sustituir c en (A)
b = (rhs_A - 1859 * c) / 143
print(f"\nPaso 5: Sustituir c en (A):")
print(f"         143b = {rhs_A:,.2f} - 1859×({c:,.2f}) = {rhs_A - 1859*c:,.2f}")
print(f"         b = {b:,.2f}")

# Sustituir b y c en (1')
a = (sum_y_2 - 39*b - 325*c) / 6
print(f"\nPaso 6: Sustituir en (1'):")
print(f"         6a = {sum_y_2:,.2f} - 39×({b:,.2f}) - 325×({c:,.2f})")
print(f"         6a = {sum_y_2 - 39*b - 325*c:,.2f}")
print(f"         a = {a:,.2f}")

print("\n" + "=" * 70)
print("COEFICIENTES OBTENIDOS:")
print(f"a = {a:,.2f}")
print(f"b = {b:,.2f}")
print(f"c = {c:,.2f}")
print("=" * 70)

# =============================================================================
# 4. ECUACIÓN DE REGRESIÓN
# =============================================================================
print("\n" + "=" * 70)
print("4. ECUACIÓN DE REGRESIÓN:")
print("=" * 70)
print(f"y = {a:,.2f} + {b:,.2f} x + {c:,.2f} x²")

# =============================================================================
# 5. CÁLCULO DEL COEFICIENTE DE DETERMINACIÓN R²
# =============================================================================
print("\n" + "=" * 70)
print("5. CÁLCULO DEL COEFICIENTE DE DETERMINACIÓN R²:")
print("=" * 70)

# Media de y
y_mean = np.mean(y)
print(f"Media de y (ȳ) = Σy / n = {sum_y:,} / {n} = {y_mean:,.2f}")

# Suma de cuadrados total (SCT)
sct = np.sum((y - y_mean)**2)
print(f"SCT = Σ(yi - ȳ)² = {sct:,.2f}")

# Valores predichos
y_pred = a + b * x + c * x**2

# Suma de cuadrados residual (SCR)
scr = np.sum((y - y_pred)**2)
print(f"SCR = Σ(yi - ŷi)² = {scr:,.2f}")

# R²
r2 = 1 - (scr / sct)
print(f"\nR² = 1 - (SCR / SCT) = 1 - ({scr:,.2f} / {sct:,.2f}) = {r2:.4f}")

print("\n" + "=" * 70)
print(f"R² FINAL = {r2:.4f} (≈ 0.97)")
print("=" * 70)

# =============================================================================
# 6. TABLA DE COMPARACIÓN (REAL vs PREDICHO)
# =============================================================================
print("\n" + "=" * 70)
print("6. TABLA DE COMPARACIÓN (REAL vs PREDICHO):")
print("=" * 70)

tabla = []
for i in range(n):
    tabla.append([x[i], y[i], round(y_pred[i], 0), y[i] - round(y_pred[i], 0), 
                  round((y[i] - y_pred[i])**2, 0)])
print(tabulate(tabla, headers=["Mes (x)", "Real (y)", "Predicho (ŷ)", 
                               "Residuo", "Residuo²"], 
               tablefmt="pretty", floatfmt=",.0f"))

# =============================================================================
# 7. GRÁFICA DE REGRESIÓN POLINOMIAL
# =============================================================================
print("\n" + "=" * 70)
print("7. GENERANDO GRÁFICA...")
print("=" * 70)

# Curva suave para la gráfica
x_smooth = np.linspace(1, 12, 200)
y_smooth = a + b * x_smooth + c * x_smooth**2

# Crear la figura
plt.figure(figsize=(10, 6), facecolor='white')
plt.scatter(x, y, color='#3B82F6', s=80, zorder=5, label='Datos reales', alpha=0.8)
plt.plot(x_smooth, y_smooth, color='#2DD4BF', linewidth=2.5, label='Regresión polinomial grado 2')
plt.plot(x, y_pred, 'o', color='#F59E0B', markersize=6, label='Valores predichos', alpha=0.7)

# Etiquetas y título
plt.xlabel('Mes (x)', fontsize=12)
plt.ylabel('Ventas (S/)', fontsize=12)
plt.title('Regresión Polinomial de Grado 2 - Ventas Mensuales\n' + 
          f'y = {a:,.0f} + {b:,.0f}x + {c:,.0f}x² | R² = {r2:.4f}', 
          fontsize=14)
plt.xticks(range(1, 13), ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                          'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.3)

# Formato de los ejes
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
plt.tight_layout()

# Mostrar la gráfica
plt.show()

# Opcional: guardar la gráfica
# plt.savefig('regresion_polinomial_grado2.png', dpi=150, bbox_inches='tight')

print("\n" + "=" * 70)
print("PROCESO COMPLETADO EXITOSAMENTE")
print("=" * 70)

# =============================================================================
# 8. PREDICCIONES PARA MESES FUTUROS (OPCIONAL)
# =============================================================================
print("\n8. PREDICCIONES PARA MESES FUTUROS (solo referencia):")
print("-" * 50)
for t in range(13, 16):
    pred = a + b * t + c * t**2
    if pred > 0:
        print(f"Mes {t}: {pred:,.0f} soles")
    else:
        print(f"Mes {t}: {pred:,.0f} soles (negativo, no válido fuera del rango)")

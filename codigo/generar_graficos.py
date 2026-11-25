import pandas as pd
import matplotlib.pyplot as plt

# Configuración estética académica
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "Arial"

# Cargar el dataset limpio
df = pd.read_csv("resultados_sensibilidad_k_RESUMEN.csv")
df_5000 = df[df["N"] == 5000] # Nos enfocamos en la red grande bajo estrés

# Figura 1: Resiliencia (Recall vs Churn)
plt.figure(figsize=(8, 5))
for k in [3, 8, 14, 20]:
    data = df_5000[df_5000["k"] == k]
    plt.plot(data["churn_pct"], data["recall_mean"], marker='o', linewidth=2, label=f'k={k}')
plt.axhline(y=95, color='r', linestyle='--', alpha=0.7, label='SLA Mínimo (95%)')
plt.title("Tasa de Éxito (Recall) vs. Churn Masivo (N=5000)")
plt.xlabel("Porcentaje de Churn (%)")
plt.ylabel("Tasa de Éxito (%)")
plt.legend()
plt.savefig("figura1_tasa_exito_vs_churn.png", dpi=300, bbox_inches='tight')
plt.close()

# Figura 2: Overhead Operativo vs k (Churn 30%)
plt.figure(figsize=(8, 5))
data_30 = df_5000[df_5000["churn_pct"] == 30]
plt.bar(data_30["k"].astype(str), data_30["overhead_mean"], color='#2c7bb6', alpha=0.8)
plt.title("Overhead Operativo Puro bajo 30% de Churn (N=5000)")
plt.xlabel("Tamaño del Bucket (k)")
plt.ylabel("Mensajes Totales por Búsqueda")
plt.savefig("figura2_mensajes_vs_k.png", dpi=300, bbox_inches='tight')
plt.close()

# Figura 3: Saltos Promedio (Hops) vs k
plt.figure(figsize=(8, 5))
plt.plot(data_30["k"], data_30["hops_mean"], marker='s', color='#d7191c', linewidth=2)
plt.title("Saltos de Ruteo (Hops) necesarios bajo 30% de Churn (N=5000)")
plt.xlabel("Tamaño del Bucket (k)")
plt.ylabel("Cantidad de Saltos")
plt.xticks([3, 8, 14, 20])
plt.savefig("figura4_saltos_vs_k.png", dpi=300, bbox_inches='tight')
plt.close()

# Figura 4: Índice de Trade-off (Recall / Overhead)
plt.figure(figsize=(8, 5))
plt.plot(data_30["k"], data_30["tradeoff_index"], marker='D', color='#1a9641', linewidth=2)
plt.title("Índice de Trade-off (Recall / Overhead) bajo 30% Churn")
plt.xlabel("Tamaño del Bucket (k)")
plt.ylabel("Índice (Mayor es mejor)")
plt.xticks([3, 8, 14, 20])
plt.savefig("figura3_indice_tradeoff_vs_k.png", dpi=300, bbox_inches='tight')
plt.close()

print("[+] Los 4 gráficos PNG han sido generados exitosamente.")
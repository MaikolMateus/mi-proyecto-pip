# ── main.py ───────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

from utils import (
    descargar_datos, calcular_retornos,
    construir_clusters, graficar_umap,
    construir_factor_util,
    correr_garch, correr_arima, correr_arima_garch,
    evaluar_lstm_accion,
)

# ── Configuración ─────────────────────────────────────────────────────────────
N_TICKERS    = 50
START        = "2020-01-01"
END          = "2025-01-01"
EXCLUIR      = ["ABNB", "APP"]
CLUSTER_UTIL = ["AWK", "AEP", "ATO", "AEE", "LNT"]
N_CLUSTERS   = 4
TRAIN_RATIO  = 0.8

# ── 1. Datos ──────────────────────────────────────────────────────────────────
print("── 1. Datos ──────────────────────────────────────────────────────────")
cierres = descargar_datos(n=N_TICKERS, start=START, end=END, excluir=EXCLUIR)

# ── 2. Retornos ───────────────────────────────────────────────────────────────
print("── 2. Retornos ───────────────────────────────────────────────────────")
retornos = calcular_retornos(cierres)

# ── 3. Clusters + UMAP ───────────────────────────────────────────────────────
print("── 3. Clusters ───────────────────────────────────────────────────────")
features, X_features, clusters = construir_clusters(retornos, n_clusters=N_CLUSTERS)
graficar_umap(features, X_features, clusters)

# ── 4. Factor Utilities ───────────────────────────────────────────────────────
print("── 4. Factor Utilities ───────────────────────────────────────────────")
factor_util = construir_factor_util(retornos, CLUSTER_UTIL)

train_size = int(TRAIN_RATIO * len(factor_util))
train      = factor_util.iloc[:train_size]
test       = factor_util.iloc[train_size:]

# ── 5. GARCH ──────────────────────────────────────────────────────────────────
print("── 5. GARCH ──────────────────────────────────────────────────────────")
correr_garch(factor_util, train, test, train_size)

# ── 6. ARIMA ──────────────────────────────────────────────────────────────────
print("── 6. ARIMA ──────────────────────────────────────────────────────────")
p_opt, q_opt = correr_arima(factor_util, train, test)

# ── 7. ARIMA-GARCH ────────────────────────────────────────────────────────────
print("── 7. ARIMA-GARCH ────────────────────────────────────────────────────")
correr_arima_garch(factor_util, train, test, train_size, p_opt)

# ── 8. LSTM ───────────────────────────────────────────────────────────────────
print("── 8. LSTM ───────────────────────────────────────────────────────────")
resultados = []
for ticker in CLUSTER_UTIL:
    res = evaluar_lstm_accion(cierres, ticker)
    resultados.append(res)

import pandas as pd
print(pd.DataFrame(resultados).sort_values("RMSE").to_string(index=False))
print("\n✓ Pipeline completado.")

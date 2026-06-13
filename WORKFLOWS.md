# WORKFLOWS — Flujo del análisis S&P 500

---

## Bloque 1 — Ingesta de datos

```
┌─────────────────────────────────────────────────────────┐
│  FUENTE: Wikipedia + Yahoo Finance                      │
│                                                         │
│  • Scraping lista S&P 500 (Wikipedia)                   │
│  • Descarga top 50 tickers vía yfinance                 │
│  • Período: 2020-01-01 → 2025-01-01                     │
│  • Variable: Precio de cierre diario                    │
│  • Exclusiones: ABNB, APP (IPO tardía → muchos NaN)     │
│                                                         │
│  OUTPUT → cierres_limpio  shape: (~1257, 48)            │
└─────────────────────────────────────────────────────────┘
```

**Función:** `descargar_datos()` en `utils.py`

---

## Bloque 2 — Retornos logarítmicos

```
┌─────────────────────────────────────────────────────────┐
│  INPUT: cierres_limpio                                  │
│                                                         │
│  Rt = ln(Pt / Pt-1)                                     │
│                                                         │
│  ¿Por qué retornos y no precios?                        │
│  • Precios tienen tendencia y escalas distintas         │
│  • Retornos son estacionarios y comparables             │
│  • Eliminan efectos de nivel entre activos              │
│                                                         │
│  OUTPUT → retornos  shape: (~1257, 48)                  │
└─────────────────────────────────────────────────────────┘
```

**Función:** `calcular_retornos()` en `utils.py`

---

## Bloque 3 — Análisis exploratorio

```
┌───────────────────────┐    ┌───────────────────────┐
│  KDE + Normal AAPL    │    │  Detección atípicos   │
│                       │    │                       │
│  • Colas pesadas      │    │  • KDE gaussiano      │
│  • Asimetría          │    │  • Umbral percentil 1 │
│  • Fat tails          │    │  • Fechas extremas    │
└───────────────────────┘    └───────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Mapas de calor correlación                             │
│                                                         │
│  corr1: ene–jun 2021    corr2: jul–oct 2021             │
│  Δcorr = corr2 - corr1  (cambios estructurales)        │
└─────────────────────────────────────────────────────────┘
```

---

## Bloque 4 — Clustering K-Means

```
┌─────────────────────────────────────────────────────────┐
│  FEATURES por ticker (48 × 4)                           │
│                                                         │
│  mean │ std │ skew │ kurt                               │
│                                                         │
│  → StandardScaler → K-Means(k=4, seed=123)             │
└─────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│  MÉTODO DEL CODO → k=4 óptimo                           │
└──────────────────────────────────────────────────────────┘
           │
           ▼
┌────────────────┬────────────────┬────────────────────────┐
│ 🟡 TECNOLÓGICO │ 🟢 UTILITIES   │ 🔴 MIXTO               │
│                │                │                        │
│ AAPL, AMZN     │ AWK, AEP       │ ABBV, AXP, AFL         │
│ GOOG, AMD      │ ATO, AEE, LNT  │ AIG, MMM, ACN...       │
│ ADBE, ANET...  │                │ (33 tickers)           │
└────────────────┴────────────────┴────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│  VISUALIZACIÓN UMAP 2D                                  │
│  n_neighbors=15 │ min_dist=0.1 │ random_state=123       │
└─────────────────────────────────────────────────────────┘
```

**Funciones:** `construir_clusters()`, `graficar_umap()` en `utils.py`

---

## Bloque 5 — Correlaciones y PCA por cluster

```
┌─────────────────────────────────────────────────────────┐
│  CORRELACIONES                                          │
│                                                         │
│  Pearson  → lineal                                      │
│  Spearman → monotónica (más robusta a outliers)         │
│                                                         │
│  Tecnológico  Pearson: ~0.65   Spearman: ~0.68          │
│  Utilities    Pearson: ~0.55   Spearman: ~0.57          │
│  Mixto        Pearson: ~0.35   Spearman: ~0.37          │
└─────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│  PCA por cluster                                        │
│                                                         │
│  PC1 Tecnológico: ~55% varianza explicada               │
│  PC1 Utilities:   ~65% varianza explicada  ← usar esto  │
│  PC1 Mixto:       ~30% varianza explicada               │
│                                                         │
│  → factor_util = PC1 del cluster Utilities              │
│    Serie: (~1257,)  índice: fechas diarias              │
└─────────────────────────────────────────────────────────┘
```

**Función:** `construir_factor_util()` en `utils.py`

---

## Bloque 6 — GARCH (volatilidad)

```
┌─────────────────────────────────────────────────────────┐
│  PRUEBAS PREVIAS                                        │
│                                                         │
│  ADF   p=0.000 → estacionaria       ✓                   │
│  ARCH-LM p=0.000 → efectos ARCH     ✓                   │
└─────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│  SELECCIÓN POR AIC                                      │
│                                                         │
│  GARCH(1,1)-t     AIC: 3953 ✓  ← ganador               │
│  GARCH(1,1)-Normal AIC: 3979                            │
│  EGARCH(1,1)       AIC: 3991                            │
└─────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│  PARÁMETROS  GARCH(1,1)-t                               │
│                                                         │
│  ω = 0.253  (nivel base)                                │
│  α = 0.134  (impacto shocks recientes)                  │
│  β = 0.793  (persistencia)                              │
│  α + β = 0.927 → alta persistencia, estacionario        │
│  ν = 7.74   (colas pesadas — t justificada)             │
│                                                         │
│  Ljung-Box p=0.82/0.90 → residuos limpios  ✓           │
└─────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│  WALK-FORWARD ventana 1 año → predice mes a mes         │
│                                                         │
│  ene 2023 – dic 2023  →  predice ene 2024               │
│  ene 2023 – ene 2024  →  predice feb 2024               │
│  ...                                                    │
│  ene 2023 – nov 2024  →  predice dic 2024               │
│                                                         │
│  Modelo: GJR-GARCH(1,1) skewt (captura asimetría)      │
└─────────────────────────────────────────────────────────┘
```

**Función:** `correr_garch()` en `utils.py`

---

## Bloque 7 — ARIMA (media)

```
┌─────────────────────────────────────────────────────────┐
│  IDENTIFICACIÓN                                         │
│                                                         │
│  ACF / PACF sobre train → d=0 (ya estacionaria)        │
│  Grid search p,q ∈ {0,1,2,3} → selección por AIC       │
└─────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│  WALK-FORWARD MENSUAL — 1 predicción por mes            │
│                                                         │
│  Train: 2020-01-01 → mes anterior                       │
│  Pred:  promedio del mes siguiente                      │
│                                                         │
│  Serie suavizada EWM(span=5) → más estructura           │
│  Anclaje al último valor real + tendencia del modelo    │
│                                                         │
│  OUTPUT: 12 puntos (uno por mes del 2024)               │
└─────────────────────────────────────────────────────────┘

  NOTA: ARIMA predice la MEDIA (~0 en retornos financieros)
        El valor está en el IC — qué tan incierto es el mes
        El GARCH complementa con la VOLATILIDAD del período
```

**Función:** `correr_arima()` en `utils.py`

---

## Bloque 8 — ARIMA-GARCH combinado

```
┌─────────────────────────────────────────────────────────┐
│  MODELO COMBINADO                                       │
│                                                         │
│  y_t  = μ + AR(p) + ε_t          ← ARIMA (media)       │
│  ε_t  = σ_t · z_t                                      │
│  σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}  ← GARCH (var)   │
│  z_t  ~ t(ν)                                           │
└─────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│  IMPLEMENTACIÓN                                         │
│                                                         │
│  arch_model(mean='ARX', lags=p_opt,                     │
│             vol='GARCH', p=1, q=1, dist='t')            │
│                                                         │
│  Walk-forward con warm-start de parámetros              │
│  → 3-5x más rápido que sin warm-start                   │
└─────────────────────────────────────────────────────────┘
```

**Función:** `correr_arima_garch()` en `utils.py`

---

## Bloque 9 — LSTM

```
┌─────────────────────────────────────────────────────────┐
│  SERIE: Precios originales por ticker                   │
│  Escalamiento: MinMaxScaler → [0, 1]                    │
│  Ventana de entrada: 60 días                            │
│  Split: 80% train / 20% test                            │
└─────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│  ARQUITECTURA                                           │
│                                                         │
│  Input  (60, 1)                                         │
│    │                                                    │
│  LSTM(64, return_sequences=True)                        │
│    │                                                    │
│  Dropout(0.2)                                           │
│    │                                                    │
│  LSTM(32)                                               │
│    │                                                    │
│  Dropout(0.2)                                           │
│    │                                                    │
│  Dense(1)   ← predicción del día siguiente              │
└─────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│  ENTRENAMIENTO                                          │
│                                                         │
│  optimizer: Adam   loss: MSE                            │
│  epochs: 100       batch_size: 32                       │
│  EarlyStopping(patience=10, restore_best_weights=True)  │
│  validation_split: 0.2                                  │
└─────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│  EVALUACIÓN por ticker del cluster Utilities            │
│                                                         │
│  AWK │ AEP │ ATO │ AEE │ LNT                            │
│                                                         │
│  Métricas: RMSE / MAE sobre precios originales          │
└─────────────────────────────────────────────────────────┘
```

**Función:** `evaluar_lstm_accion()` en `utils.py`

---

## Pipeline completo

```
Wikipedia / Yahoo Finance
          │
          ▼
  descargar_datos()
          │
          ▼
  calcular_retornos()
          │
    ┌─────┴──────┐
    ▼            ▼
EDA / KDE    construir_clusters()
             │
             ├── correlaciones Pearson/Spearman
             ├── PCA por cluster
             └── graficar_umap()
                      │
                      ▼
          construir_factor_util()  → factor_util (PC1)
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
    correr_garch() correr_arima() correr_arima_garch()
          │           │           │
          └───────────┴───────────┘
                      │
                      ▼
          evaluar_lstm_accion()
                      │
                      ▼
          Métricas: RMSE / MAE
```

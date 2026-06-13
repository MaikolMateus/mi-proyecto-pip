# DATABASE — Descripción de los datos

---

## Fuente

| Campo | Detalle |
|---|---|
| **Proveedor** | Yahoo Finance vía `yfinance` |
| **Lista tickers** | Wikipedia — List of S&P 500 companies |
| **Período** | 2020-01-01 → 2025-01-01 |
| **Frecuencia** | Diaria (días hábiles ~1257 obs) |
| **Variable** | Precio de cierre ajustado (`Close`) |

---

## Tickers excluidos

| Ticker | Razón | IPO |
|---|---|---|
| ABNB | Muchos NaN | 09/12/2020 |
| APP | Muchos NaN | 2021 |

**Tickers finales: 48**

---

## Clusters

### 🟡 Tecnológico (10 tickers)
`ADI, AMAT, AAPL, GOOG, GOOGL, AKAM, ADBE, AMZN, AMD, ANET`

### 🟢 Utilities (5 tickers) — base del factor_util
| Ticker | Empresa | Servicio |
|---|---|---|
| AWK | American Water Works | Agua |
| AEP | American Electric Power | Electricidad |
| ATO | Atmos Energy | Gas natural |
| AEE | Ameren Corporation | Electricidad/Gas |
| LNT | Alliant Energy | Electricidad/Gas |

### 🔴 Mixto (33 tickers)
`ABBV, AMGN, ADM, MO, T, APA, AES, APTV, ALB, ALGN, AXP, AFL, AIG,
AIZ, ACGL, ALL, APO, ARES, AME, AMP, ACN, APH, AMT, ARE, A, ABT,
AJG, AON, AOS, MMM, APD, ALLE, AMCR`

---

## Variables derivadas

| Variable | Shape | Descripción |
|---|---|---|
| `cierres_limpio` | (~1257, 48) | Precios de cierre |
| `retornos` | (~1257, 48) | Retornos log diarios |
| `features` | (48, 4) | mean, std, skew, kurt por ticker |
| `factor_util` | (~1257,) | PC1 cluster Utilities |

### Partición temporal

| Partición | Obs | Período |
|---|---|---|
| Train GARCH/ARIMA (80%) | ~1005 | 2020-01 → 2023-12 |
| Test GARCH/ARIMA (20%) | ~252 | 2024-01 → 2024-12 |
| Train LSTM (80%) | varía | por ticker |
| Test LSTM (20%) | varía | por ticker |

---

## Archivo generado

| Archivo | Descripción |
|---|---|
| `sp500_historico.csv` | Precios de cierre (no versionado en Git) |

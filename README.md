# Análisis S&P 500 — Clusters, GARCH, ARIMA y LSTM

Pipeline de análisis financiero sobre las primeras 50 acciones del S&P 500.
Cubre desde la descarga de datos hasta modelos de predicción con redes neuronales.

---

## Estructura del proyecto

```
mi-proyecto-pip/
├── main.py          # Pipeline principal
├── utils.py         # Funciones reutilizables
├── requirements.txt # Dependencias
├── .env.example     # Variables de entorno
├── .gitignore
├── README.md
├── WORKFLOWS.md     # Flujo visual del análisis (bloques)
└── DATABASE.md      # Descripción de los datos
```

---

## Instalación

```bash
git clone https://github.com/tu-usuario/mi-proyecto-pip.git
cd mi-proyecto-pip

python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

pip install -r requirements.txt

cp .env.example .env
```

---

## Uso

```bash
python main.py
```

---

## Modelos implementados

| Modelo | Objetivo | Método |
|---|---|---|
| K-Means + UMAP | Clustering de acciones | Features estadísticas |
| GARCH(1,1)-t | Volatilidad condicional | Walk-forward mensual |
| ARIMA | Media de la serie | Walk-forward mensual |
| ARIMA-GARCH | Media + Volatilidad | Walk-forward warm-start |
| LSTM | Precio original | Ventana 60 días |

---

## Resultados principales

| Cluster | Sector | PC1 varianza explicada |
|---|---|---|
| Tecnológico | Semiconductores, Software | ~55% |
| **Utilities** | Agua, Electricidad, Gas | **~65%** ← modelo |
| Mixto | Industriales, Finanzas, Salud | ~30% |

| Modelo | RMSE | MAE |
|---|---|---|
| GARCH(1,1)-t | — | — |
| ARIMA | — | — |
| ARIMA-GARCH | — | — |
| LSTM | — | — |

> Completa con los valores al correr el pipeline.

---

## Dependencias principales

```
yfinance · arch · statsmodels · scikit-learn · tensorflow · umap-learn
```

#  Predicción de Volatilidad en el S&P 500 mediante PCA, Clustering, ARIMA-GARCH y LSTM

Este proyecto desarrolla una metodología para el análisis y predicción de volatilidad en acciones del índice S&P 500 utilizando técnicas de reducción de dimensión, aprendizaje no supervisado, modelos econométricos y redes neuronales profundas.

El objetivo principal es identificar grupos homogéneos de activos financieros, extraer factores latentes mediante Análisis de Componentes Principales (PCA) y comparar el desempeño predictivo de modelos tradicionales de volatilidad frente a modelos de Deep Learning.

---

# Objetivos

- Descargar precios históricos de acciones del S&P 500.
- Transformar los precios en rendimientos logarítmicos.
- Detectar grupos homogéneos mediante K-Means.
- Analizar la estructura de correlaciones entre activos.
- Construir factores latentes mediante PCA.
- Modelar la volatilidad utilizando modelos GARCH.
- Comparar modelos econométricos y redes neuronales LSTM.
- Evaluar el desempeño predictivo mediante métricas de error.

---

# Conjunto de Datos

**Fuente:** Yahoo Finance

**Periodo de estudio:**

- Enero de 2020
- Enero de 2025

**Activos analizados:**

- Primeras 50 empresas pertenecientes al índice S&P 500.

**Activos excluidos:**

- ABNB
- APP

Debido a la gran cantidad de valores faltantes ocasionados por su fecha reciente de incorporación al mercado.

---

# Metodología

## 1. Descarga de datos

Obtención automática de precios ajustados desde Yahoo Finance.

## 2. Limpieza de datos

- Eliminación de activos con valores faltantes.
- Verificación de consistencia temporal.

## 3. Cálculo de rendimientos

Se calcularon rendimientos logarítmicos mediante:

\[
r_t=\ln\left(\frac{P_t}{P_{t-1}}\right)
\]

## 4. Estandarización

Los rendimientos fueron escalados utilizando:

- StandardScaler

para evitar sesgos asociados a diferencias de magnitud entre activos.

## 5. Detección de valores atípicos

Se realizó un análisis exploratorio utilizando:

- Kernel Density Estimation (KDE)

sobre activos representativos.

## 6. Análisis de correlación

Se construyeron matrices de correlación para identificar relaciones lineales entre activos.

## 7. Clustering

Se aplicó:

- K-Means

para identificar grupos homogéneos de empresas según el comportamiento de sus rendimientos.

## 8. Reducción de dimensión

Se utilizaron:

- PCA
- UMAP

para visualizar la estructura de los clusters obtenidos.

## 9. Selección del cluster Utilities

El cluster seleccionado estuvo compuesto por:

- AWK
- AEP
- ATO
- AEE
- LNT

Este grupo presentó:

- Alta correlación interna.
- Comportamiento homogéneo.
- Mayor presencia de un factor común.

## 10. Construcción del factor principal

Se aplicó PCA al cluster Utilities y se utilizó la primera componente principal (PC1) como factor representativo.

## 11. Diagnósticos estadísticos

Se realizaron:

- Prueba Dickey-Fuller Aumentada (ADF).
- Prueba ARCH-LM.

para verificar estacionariedad y heterocedasticidad condicional.

## 12. Modelación de volatilidad

Se ajustaron los siguientes modelos:

### Modelos econométricos

- GARCH(1,1)
- GARCH(2,1)
- GARCH(1,2)
- EGARCH(1,1)
- ARIMA
- ARIMA-GARCH

### Modelos de Deep Learning

- LSTM sobre el factor PCA.
- LSTM sobre precios originales.

---

# Estructura del Proyecto

```text
sp500-volatility/
│
├── data/
│   └── Datos descargados
│
├── notebooks/
│   └── Análisis exploratorio
│
├── src/
│   ├── descarga_datos.py
│   ├── clustering.py
│   ├── pca.py
│   ├── garch.py
│   └── lstm.py
│
├── results/
│   └── Resultados y métricas
│
├── figures/
│   └── Gráficos y visualizaciones
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# Tecnologías Utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Statsmodels
- ARCH
- TensorFlow
- Keras
- UMAP

---

# Resultados

Los modelos fueron comparados utilizando:

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

| Modelo | 
|----------|
| GARCH | 
| ARIMA-GARCH | 
| LSTM | 

El mejor modelo será seleccionado con base en su capacidad predictiva sobre datos no observados.

---

# Flujo General del Proyecto

```text
Precios S&P 500
        │
        ▼
Rendimientos Logarítmicos
        │
        ▼
Estandarización
        │
        ▼
Análisis de Correlación
        │
        ▼
K-Means Clustering
        │
        ▼
Cluster Utilities
        │
        ▼
PCA (PC1)
        │
        ▼
 ┌──────────────────┬─────────────────┐
 │   ARIMA-GARCH    │      LSTM       │
 └──────────────────┴─────────────────┘
        │
        ▼
Comparación de Desempeño
```

---

# Referencias

- Engle, R. F. (1982). Autoregressive Conditional Heteroskedasticity.
- Bollerslev, T. (1986). Generalized ARCH Models.
- Jolliffe, I. (2002). Principal Component Analysis.
- Box, Jenkins y Reinsel (2015). Time Series Analysis.
- Hochreiter y Schmidhuber (1997). Long Short-Term Memory.

---

# Autor

**Maikol Mateus Lucuara**

Estudiante de Matemáticas

Universidad Nacional de Colombia

Proyecto desarrollado para análisis financiero, series de tiempo y aprendizaje automático aplicado a mercados financieros.

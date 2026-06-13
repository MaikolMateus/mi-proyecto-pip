# ── utils.py ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import requests
import warnings
warnings.filterwarnings("ignore")

from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns

import yfinance as yf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATOS
# ══════════════════════════════════════════════════════════════════════════════

def descargar_datos(n=50, start="2020-01-01", end="2025-01-01",
                    excluir=None) -> pd.DataFrame:
    url     = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    sp500   = pd.read_html(StringIO(requests.get(url, headers=headers).text))[0]
    tickers = [x.replace(".", "-") for x in sp500["Symbol"].tolist()]
    datos   = yf.download(tickers[:n], start=start, end=end, progress=False)
    cierres = datos["Close"]
    if excluir:
        cierres = cierres.drop(columns=[c for c in excluir if c in cierres.columns])
    cierres.to_csv("sp500_historico.csv")
    print(f"   Tickers: {cierres.shape[1]}  |  Fechas: {cierres.shape[0]}")
    return cierres


# ══════════════════════════════════════════════════════════════════════════════
# 2. RETORNOS
# ══════════════════════════════════════════════════════════════════════════════

def calcular_retornos(cierres: pd.DataFrame) -> pd.DataFrame:
    retornos = np.log(cierres / cierres.shift(1)).dropna()
    print(f"   Shape: {retornos.shape}  |  NaN: {retornos.isna().sum().sum()}")
    return retornos


# ══════════════════════════════════════════════════════════════════════════════
# 3. CLUSTERS
# ══════════════════════════════════════════════════════════════════════════════

def construir_clusters(retornos, n_clusters=4):
    features   = pd.DataFrame({
        "mean": retornos.mean(), "std": retornos.std(),
        "skew": retornos.skew(), "kurt": retornos.kurt(),
    })
    X_features = StandardScaler().fit_transform(features)
    clusters   = KMeans(n_clusters=n_clusters, random_state=123,
                        n_init=10).fit_predict(X_features)
    resultado  = pd.DataFrame({"Ticker": retornos.columns,
                                "Cluster": clusters}).sort_values("Cluster")
    print(resultado.to_string(index=False))
    return features, X_features, clusters


def graficar_umap(features, X_features, clusters):
    try:
        import umap
    except ImportError:
        print("   umap-learn no instalado."); return

    reducer = umap.UMAP(n_components=2, n_neighbors=15,
                        min_dist=0.1, random_state=123)
    X_umap  = reducer.fit_transform(X_features)

    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=clusters,
                         cmap="viridis", s=100, alpha=0.8,
                         edgecolors="white", linewidths=0.5)
    for i, ticker in enumerate(features.index):
        ax.annotate(ticker, xy=(X_umap[i, 0], X_umap[i, 1]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    fig.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP de acciones S&P 500 coloreado por clusters K-Means")
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 4. FACTOR UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def construir_factor_util(retornos, cluster_util) -> pd.Series:
    X   = StandardScaler().fit_transform(retornos[cluster_util])
    pca = PCA(n_components=1)
    factor_util = pd.Series(pca.fit_transform(X)[:, 0],
                            index=retornos.index, name="factor_util")
    plt.figure(figsize=(12, 4))
    plt.plot(factor_util)
    plt.title("PC1 del Cluster Utilities")
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
    return factor_util


# ══════════════════════════════════════════════════════════════════════════════
# 5. GARCH — walk-forward mensual
# ══════════════════════════════════════════════════════════════════════════════

def correr_garch(factor_util, train, test, train_size):
    # Pruebas
    adf_stat, adf_pval, *_ = adfuller(factor_util)
    arch_stat, arch_pval, _, _ = het_arch(factor_util)
    print(f"   ADF p={adf_pval:.4f} | ARCH-LM p={arch_pval:.4f}")

    # Selección de modelo
    specs = {
        "GARCH(1,1)-Normal": arch_model(train, mean="Constant", vol="GARCH", p=1, q=1, dist="normal"),
        "GARCH(1,1)-t":      arch_model(train, mean="Constant", vol="GARCH", p=1, q=1, dist="t"),
        "GJR-GARCH(1,1)-t":  arch_model(train, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t"),
        "EGARCH(1,1)":       arch_model(train, mean="Constant", vol="EGARCH", p=1, q=1),
    }
    resultados = {}
    print(f"\n   {'Modelo':<22} {'AIC':>10} {'BIC':>10}")
    print("   " + "-" * 44)
    for nombre, modelo in specs.items():
        res = modelo.fit(disp="off")
        resultados[nombre] = res
        print(f"   {nombre:<22} {res.aic:>10.2f} {res.bic:>10.2f}")

    mejor     = min(resultados, key=lambda k: resultados[k].aic)
    res_final = resultados[mejor]
    print(f"\n   → Mejor: {mejor}")
    print(res_final.summary())

    lb = acorr_ljungbox(res_final.std_resid**2, lags=[10, 20], return_df=True)
    print("\n   Ljung-Box:"); print(lb)

    # Walk-forward mensual
    INICIO_TRAIN = factor_util.index[0].strftime("%Y-%m-%d")
    INICIO_TEST  = test.index[0].strftime("%Y-%m-%d")
    meses_test   = pd.date_range(start=INICIO_TEST,
                                 end=factor_util.index[-1], freq="MS")

    pred_vol_wf = []; fechas_pred = []
    print(f"\n   Walk-forward mensual ({len(meses_test)} meses)...")

    for mes in meses_test:
        fin_mes = mes + pd.offsets.MonthEnd(1)
        test_i  = factor_util.loc[mes:fin_mes]
        if len(test_i) == 0: continue

        for j in range(len(test_i)):
            train_j = factor_util.loc[INICIO_TRAIN:
                                      test_i.index[j] - pd.offsets.BDay(1)]
            if len(train_j) < 20: continue
            res_j = arch_model(train_j, mean="Constant", vol="GARCH",
                               p=1, o=1, q=1, dist="skewt").fit(disp="off")
            vol_j = np.sqrt(res_j.forecast(horizon=1).variance.iloc[-1, 0])

            resid_rec  = train_j.iloc[-5:].std()
            vol_media  = pred_vol_wf[-20:] if len(pred_vol_wf) >= 20 else [vol_j]
            factor_esc = np.clip(resid_rec / np.mean(vol_media), 0.7, 1.5)
            pred_vol_wf.append(vol_j * factor_esc)
            fechas_pred.append(test_i.index[j])

        print(f"   ✓ {mes.strftime('%Y-%m')}")

    pred_vol_s = pd.Series(pred_vol_wf, index=fechas_pred)
    vol_real   = test.rolling(20).std().dropna()
    pred_alin  = pred_vol_s.loc[vol_real.index]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(vol_real.index,  vol_real,  label="Vol. realizada (rolling 20d)", lw=1.5)
    ax.plot(pred_alin.index, pred_alin, label="GJR-GARCH (walk-forward)",     lw=1.5, ls="--")
    for mes in meses_test:
        ax.axvline(mes, color="gray", ls=":", lw=0.6, alpha=0.4)
    ax.set_title("GJR-GARCH — Volatilidad walk-forward mensual  |  PC1 Utilities")
    ax.set_xlabel("Fecha"); ax.set_ylabel("Volatilidad")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

    rmse = np.sqrt(mean_squared_error(vol_real, pred_alin))
    mae  = mean_absolute_error(vol_real, pred_alin)
    print(f"\n   RMSE GARCH: {rmse:.6f}  |  MAE: {mae:.6f}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. ARIMA — walk-forward mensual anclado
# ══════════════════════════════════════════════════════════════════════════════

def correr_arima(factor_util, train, test) -> tuple:
    # ACF / PACF
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    plot_acf(train,  lags=30, ax=axes[0], title="ACF — factor_util (train)")
    plot_pacf(train, lags=30, ax=axes[1], title="PACF — factor_util (train)")
    plt.tight_layout(); plt.show()

    # Selección de orden
    resultados_arima = {}
    for p in range(0, 4):
        for q in range(0, 4):
            try:
                res = ARIMA(train, order=(p, 0, q)).fit()
                resultados_arima[(p, q)] = (res.aic, res.bic, res)
            except Exception:
                continue

    tabla = pd.DataFrame(
        {f"ARIMA({p},{q})": {"AIC": aic, "BIC": bic}
         for (p, q), (aic, bic, _) in resultados_arima.items()}
    ).T.sort_values("AIC")
    print(tabla.head(8).round(2))

    mejor_orden = tabla.index[0]
    p_opt, q_opt = [int(x) for x in
                    mejor_orden.replace("ARIMA(", "").replace(")", "").split(",")]
    print(f"\n   → {mejor_orden}  p={p_opt} q={q_opt}")

    # Walk-forward mensual anclado
    INICIO_TRAIN = "2020-01-01"
    INICIO_TEST  = test.index[0].strftime("%Y-%m-%d")
    factor_suav  = factor_util.ewm(span=5, adjust=False).mean()
    meses_test   = pd.date_range(start=INICIO_TEST,
                                 end=factor_util.index[-1], freq="MS")

    pred_media = []; pred_ic_sup = []; pred_ic_inf = []
    fechas_pred = []; real_mensual = []

    for mes in meses_test:
        train_i  = factor_suav.loc[INICIO_TRAIN:mes - pd.offsets.BDay(1)]
        fin_mes  = mes + pd.offsets.MonthEnd(1)
        real_mes = factor_util.loc[mes:fin_mes].mean()
        if len(train_i) < 20 or pd.isna(real_mes): continue

        res_i  = ARIMA(train_i, order=(p_opt, 0, q_opt)).fit()
        n_dias = len(factor_util.loc[mes:fin_mes])
        fc     = res_i.get_forecast(steps=n_dias)

        ultimo_real = factor_util.loc[:mes - pd.offsets.BDay(1)].iloc[-1]
        pred_raw    = fc.predicted_mean.mean()
        tendencia   = pred_raw - factor_suav.loc[:mes - pd.offsets.BDay(1)].iloc[-1]
        pred_anc    = ultimo_real + tendencia

        ic = fc.conf_int(alpha=0.05)
        ancho = (ic.iloc[:, 1] - ic.iloc[:, 0]).mean() / 2

        pred_media.append(pred_anc)
        pred_ic_sup.append(pred_anc + ancho)
        pred_ic_inf.append(pred_anc - ancho)
        fechas_pred.append(mes)
        real_mensual.append(real_mes)
        print(f"   ✓ {mes.strftime('%Y-%m')}  pred={pred_anc:.4f}  real={real_mes:.4f}")

    pred_s   = pd.Series(pred_media,   index=fechas_pred)
    real_s   = pd.Series(real_mensual, index=fechas_pred)
    ic_sup_s = pd.Series(pred_ic_sup,  index=fechas_pred)
    ic_inf_s = pd.Series(pred_ic_inf,  index=fechas_pred)
    train_m  = factor_util.loc[INICIO_TRAIN:INICIO_TEST].resample("MS").mean()

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(train_m.index, train_m, color="steelblue", lw=1.5, alpha=0.6,
            label="Serie train (promedio mensual)")
    ax.axvline(pd.Timestamp(INICIO_TEST), color="black", ls="--", lw=1.5,
               label="Inicio predicción")
    ax.plot(real_s.index, real_s, color="steelblue", lw=2, marker="o",
            label="Real test")
    ax.plot(pred_s.index, pred_s, color="darkorange", lw=2, marker="o",
            ls="--", label="ARIMA (1 pred/mes)")
    ax.fill_between(pred_s.index, ic_inf_s, ic_sup_s,
                    alpha=0.15, color="darkorange", label="IC 95%")
    ax.axvspan(pd.Timestamp(INICIO_TRAIN), pd.Timestamp(INICIO_TEST),
               alpha=0.04, color="steelblue")
    ax.axvspan(pd.Timestamp(INICIO_TEST), factor_util.index[-1],
               alpha=0.04, color="darkorange")
    ax.set_title("ARIMA — Serie completa + Reentrenamiento mensual  |  PC1 Utilities")
    ax.set_xlabel("Fecha"); ax.set_ylabel("factor_util")
    ax.legend(loc="upper left", fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

    rmse = np.sqrt(mean_squared_error(real_s, pred_s))
    mae  = mean_absolute_error(real_s, pred_s)
    print(f"\n   RMSE ARIMA: {rmse:.6f}  |  MAE: {mae:.6f}")
    return p_opt, q_opt


# ══════════════════════════════════════════════════════════════════════════════
# 7. ARIMA-GARCH
# ══════════════════════════════════════════════════════════════════════════════

def correr_arima_garch(factor_util, train, test, train_size, p_opt):
    modelo_ag = arch_model(train, mean="ARX", lags=p_opt,
                           vol="GARCH", p=1, q=1, dist="t")
    res_ag = modelo_ag.fit(disp="off")
    print(res_ag.summary())

    lb = acorr_ljungbox(res_ag.std_resid**2, lags=[10, 20], return_df=True)
    print("\n   Ljung-Box:"); print(lb)

    pred_vol_wf = []; params = res_ag.params
    print(f"\n   Walk-forward ({len(test)} pasos)...")

    for i in range(len(test)):
        train_i = factor_util.iloc[:train_size + i]
        res_i   = arch_model(train_i, mean="ARX", lags=p_opt,
                             vol="GARCH", p=1, q=1, dist="t"
                             ).fit(disp="off", starting_values=params)
        pred_vol_wf.append(np.sqrt(
            res_i.forecast(horizon=1).variance.iloc[-1, 0]))
        params = res_i.params

    pred_vol_s = pd.Series(pred_vol_wf, index=test.index)
    vol_real   = test.rolling(20).std().dropna()
    pred_alin  = pred_vol_s.loc[vol_real.index]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(vol_real.index,  vol_real,  label="Vol. realizada (rolling 20d)", lw=1.5)
    ax.plot(pred_alin.index, pred_alin, label="Vol. ARIMA-GARCH (walk-forward)", lw=1.5, ls="--")
    ax.set_title("ARIMA-GARCH — Volatilidad walk-forward  |  PC1 Utilities")
    ax.set_xlabel("Fecha"); ax.set_ylabel("Volatilidad")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

    rmse = np.sqrt(mean_squared_error(vol_real, pred_alin))
    mae  = mean_absolute_error(vol_real, pred_alin)
    print(f"\n   RMSE: {rmse:.6f}  |  MAE: {mae:.6f}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. LSTM
# ══════════════════════════════════════════════════════════════════════════════

def evaluar_lstm_accion(cierres, ticker, ventana=60) -> dict:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    serie      = cierres[ticker].dropna()
    train_size = int(len(serie) * 0.8)
    train      = serie.iloc[:train_size]
    test       = serie.iloc[train_size:]

    scaler       = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    test_scaled  = scaler.transform(test.values.reshape(-1, 1))

    def crear_secuencias(data, v):
        X, y = [], []
        for i in range(v, len(data)):
            X.append(data[i-v:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = crear_secuencias(train_scaled, ventana)
    datos_test       = np.concatenate([train_scaled[-ventana:], test_scaled])
    X_test,  y_test  = crear_secuencias(datos_test, ventana)

    X_train = X_train.reshape(*X_train.shape, 1)
    X_test  = X_test.reshape(*X_test.shape, 1)

    modelo = Sequential([
        LSTM(64, return_sequences=True, input_shape=(ventana, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    modelo.compile(optimizer="adam", loss="mse")

    modelo.fit(X_train, y_train, epochs=100, batch_size=32,
               validation_split=0.1, verbose=0,
               callbacks=[EarlyStopping(monitor="val_loss", patience=10,
                                        restore_best_weights=True)])

    pred   = scaler.inverse_transform(modelo.predict(X_test, verbose=0))
    y_real = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(y_real, pred))
    mae  = mean_absolute_error(y_real, pred)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test.index, y_real, label="Real")
    ax.plot(test.index, pred,   label="LSTM", ls="--")
    ax.set_title(f"LSTM — {ticker}  |  RMSE={rmse:.4f}  MAE={mae:.4f}")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

    print(f"   {ticker}  RMSE={rmse:.4f}  MAE={mae:.4f}")
    return {"Ticker": ticker, "RMSE": rmse, "MAE": mae}

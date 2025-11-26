# ===== Streamlit App - Predizione Resistenza a Compressione (RC) =====
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import io  # per il buffer immagini
from mpl_toolkits.mplot3d import Axes3D  # necessario per la superficie 3D

# === CONFIGURAZIONE PAGINA ===
st.set_page_config(page_title="Predizione Resistenza a Compressione", layout="wide")

# === STILE GRAFICI "DA ARTICOLO" ===
plt.rcParams.update({
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8
})

# === TITOLO SINTETICO ===
st.title("Predizione della resistenza a compressione (RC)")

# === CARICAMENTO DATI ===
df = pd.read_excel("sample_data/Provini Z densità completa-1.xlsx")
X_train = df[["IT", "D"]].values
y_train = df["RC"].values

IT_min, IT_max = df["IT"].min(), df["IT"].max()
D_min, D_max = df["D"].min(), df["D"].max()
RC_min, RC_max = y_train.min(), y_train.max()
RC_range = RC_max - RC_min

# === PARAMETRI GLOBALI PER L'ANALISI LOCALE (POLINOMIALE) ===
TOL_LOCAL = 0.10   # +/-10% su D per l'affidabilita' locale della polinomiale
MIN_N_LOCAL = 5    # minimo numero di provini locali per usare media/sigma

# === SIDEBAR: PARAMETRI ===
st.sidebar.header("Parametri di input")

modello_scelto = st.sidebar.radio(
    "Modello di regressione",
    ["Random Forest", "Regressione polinomiale (log–exp)"]
)

# Intervalli di addestramento SOLO per Random Forest (informativi)
if modello_scelto == "Random Forest":
    st.sidebar.markdown("**Intervalli di addestramento (dataset)**")
    st.sidebar.markdown(f"- IT: {IT_min:.2f} – {IT_max:.2f} s")
    st.sidebar.markdown(f"- D: {D_min:.1f} – {D_max:.1f} kg/m^3")
    st.sidebar.caption(
        "La Random Forest e' addestrata su questi intervalli; "
        "la regressione polinomiale puo' essere usata anche in extrapolazione con cautela."
    )

# === INPUT IT / D DIVERSI PER RF E POLINOMIALE ===
if modello_scelto == "Random Forest":
    # RF: limitiamo ai range di addestramento
    valore_IT = st.sidebar.number_input(
        "IT [s]",
        value=float(df["IT"].mean()),
        min_value=float(IT_min),
        max_value=float(IT_max),
        key="it_rf"
    )

    valore_D = st.sidebar.number_input(
        "D [kg/m^3]",
        value=float(df["D"].mean()),
        min_value=float(D_min),
        max_value=float(D_max),
        key="d_rf"
    )
else:
    # Polinomiale: nessun limite (puoi inserire qualsiasi valore)
    valore_IT = st.sidebar.number_input(
        "IT [s]",
        value=float(df["IT"].mean()),
        key="it_poly"
    )

    valore_D = st.sidebar.number_input(
        "D [kg/m^3]",
        value=float(df["D"].mean()),
        key="d_poly"
    )

if modello_scelto == "Regressione polinomiale (log–exp)":
    grado_poly = st.sidebar.slider("Grado del polinomio", 1, 3, 2)
else:
    grado_poly = 2  # placeholder, non usato in RF

IT_range = np.linspace(IT_min, IT_max, 100)

# === FUNZIONI PER L'AFFIDABILITA' ===
def calcola_affidabilita_rf(y_pred, valore_IT, valore_D):
    """Affidabilità 'globale' stile RF: posizione nel dominio (IT,D) + range RC."""
    if (
        np.isnan(y_pred)
        or np.isinf(y_pred)
        or y_pred < RC_min - 0.5 * RC_range
        or y_pred > RC_max + 0.5 * RC_range
    ):
        return "Bassa"

    IT_mean, D_mean = np.mean(X_train[:, 0]), np.mean(X_train[:, 1])
    distanza = np.sqrt(
        ((valore_IT - IT_mean) / (IT_max - IT_min)) ** 2 +
        ((valore_D - D_mean) / (D_max - D_min)) ** 2
    )

    if distanza <= 0.5:
        return "Alta"
    elif distanza <= 0.8:
        return "Media"
    else:
        return "Bassa"


def calcola_affidabilita_poly(y_pred, valore_IT, valore_D,
                              tol=TOL_LOCAL, min_n=MIN_N_LOCAL):
    """
    Affidabilità per la regressione polinomiale (log–exp).

    1) Usa la metrica locale descritta in tesi (media/sigma su RC dei provini
       con densità simile, con tolleranza fissata tol).
    2) Se i provini locali sono meno di min_n, esegue un fallback
       alla logica 'globale' tipo Random Forest (distanza in (IT, D)).
    """

    mask = (df["D"] >= valore_D * (1 - tol)) & (df["D"] <= valore_D * (1 + tol))
    y_local = df.loc[mask, "RC"].values

    if len(y_local) >= min_n:
        mean_local = np.mean(y_local)
        std_local = np.std(y_local)
        diff = abs(y_pred - mean_local)

        if diff <= std_local:
            return "Alta"
        elif diff <= 2 * std_local:
            return "Media"
        else:
            return "Bassa"

    # Fallback: pochi o zero dati locali -> criterio globale stile RF
    return calcola_affidabilita_rf(y_pred, valore_IT, valore_D)


# === LAYOUT PRINCIPALE ===
col = st.container()

col.markdown("Seleziona i parametri di input e il modello di regressione per stimare RC.")
col.caption(
    "Se i valori che vuoi stimare rientrano negli intervalli di addestramento della Random Forest, "
    "utilizza prima questo modello per ottenere una stima di riferimento e poi confronta le previsioni "
    "con la regressione polinomiale scegliendo il grado piu' opportuno."
)

# ===========================
# RANDOM FOREST
# ===========================
if modello_scelto == "Random Forest":

    col.subheader("Random Forest")

    rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict([[valore_IT, valore_D]])[0]

    # Metriche interne (le teniamo nell'expander)
    y_train_pred = rf_model.predict(X_train)
    mae = mean_absolute_error(y_train, y_train_pred)
    rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    r2 = r2_score(y_train, y_train_pred)

    with col.expander("Metriche sul set di addestramento"):
        st.write(f"MAE: {mae:.2f}")
        st.write(f"RMSE: {rmse:.2f}")
        st.write(f"R^2: {r2:.3f}")

    # === PARAMETRI E PREVISIONE (SOLO RC PREVISTO) ===
    col.markdown("### Parametri e predizione")
    col.markdown(f"**IT selezionato:** {valore_IT:.2f} s")
    col.markdown(f"**D selezionata:** {valore_D:.2f} kg/m^3")

    box_rc, _ = col.columns(2)
    box_rc.markdown(
        f"""
        <div style="border:1px solid #ccc; border-radius:5px;
                    padding:0.7rem; padding-top:0.6rem; text-align:center;">
            <div style="font-size:0.9rem; font-weight:600;">RC previsto</div>
            <div style="font-size:1.2rem; font-weight:700; margin-top:0.3rem;">
                {y_pred:.2f} MPa
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # === GRAFICO RF: ALLINEATO SOTTO RC ===
    tol_plot = 0.05
    mask_local = (df["D"] >= valore_D*(1 - tol_plot)) & (df["D"] <= valore_D*(1 + tol_plot))
    IT_local = df.loc[mask_local, "IT"].values
    RC_local = df.loc[mask_local, "RC"].values

    RC_curve = [rf_model.predict([[it, valore_D]])[0] for it in IT_range]

    col.markdown("### Diagrammi")
    col_graph = col.columns(1)[0]

    fig, ax = plt.subplots(figsize=(4.3, 2.7), dpi=130)
    if len(IT_local) > 0:
        ax.scatter(IT_local, RC_local, s=15, alpha=0.7, label="Dati locali (+/-5% D)")
    ax.plot(IT_range, RC_curve, linewidth=1.6, label="Predizione RF")
    ax.axvline(valore_IT, linestyle="--", linewidth=1)
    ax.set_xlabel("IT [s]")
    ax.set_ylabel("RC [MPa]")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    col_graph.image(buf, use_container_width=False)
    col_graph.caption(
        "Figura: andamento di RC in funzione di IT per la densità selezionata, "
        "con modello Random Forest e dati sperimentali locali."
    )

# ===========================
# REGRESSIONE POLINOMIALE
# ===========================
else:

    col.subheader("Regressione polinomiale (log–exp)")

    model = Pipeline([
        ("poly", PolynomialFeatures(degree=grado_poly, include_bias=False)),
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])

    y_log = np.log(y_train)
    model.fit(X_train, y_log)

    y_pred_log = model.predict([[valore_IT, valore_D]])[0]
    y_pred = np.exp(y_pred_log)

    affidabilita = calcola_affidabilita_poly(y_pred, valore_IT, valore_D)

    # --- Metriche sul train per il modello polinomiale (valutate su RC) ---
    y_train_log_pred = model.predict(X_train)
    y_train_pred_poly = np.exp(y_train_log_pred)
    mae_poly = mean_absolute_error(y_train, y_train_pred_poly)
    rmse_poly = np.sqrt(mean_squared_error(y_train, y_train_pred_poly))
    r2_poly = r2_score(y_train, y_train_pred_poly)

    with col.expander("Metriche sul set di addestramento"):
        st.write(f"MAE: {mae_poly:.2f}")
        st.write(f"RMSE: {rmse_poly:.2f}")
        st.write(f"R^2: {r2_poly:.3f}")

    col.markdown("### Parametri e predizione")
    col.markdown(f"**IT selezionato:** {valore_IT:.2f} s")
    col.markdown(f"**D selezionata:** {valore_D:.2f} kg/m^3")
    col.markdown(f"**Grado del polinomio:** {grado_poly}")

    box_rc, box_aff = col.columns(2)
    box_rc.markdown(
        f"""
        <div style="border:1px solid #ccc; border-radius:5px;
                    padding:0.7rem; padding-top:0.6rem; text-align:center;">
            <div style="font-size:0.9rem; font-weight:600;">RC previsto</div>
            <div style="font-size:1.2rem; font-weight:700; margin-top:0.3rem;">
                {y_pred:.2f} MPa
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    box_aff.markdown(
        f"""
        <div style="border:1px solid #ccc; border-radius:5px;
                    padding:0.7rem; padding-top:0.6rem; text-align:center;">
            <div style="font-size:0.9rem; font-weight:600;">Affidabilità</div>
            <div style="font-size:1.2rem; font-weight:700; margin-top:0.3rem;">
                {affidabilita}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # === GRAFICO DOPPIO: CURVA LOCALE + SUPERFICIE 3D RC(IT, D) ===
    mask_local = (df["D"] >= valore_D*(1 - TOL_LOCAL)) & (df["D"] <= valore_D*(1 + TOL_LOCAL))
    IT_local = df.loc[mask_local, "IT"].values
    RC_local = df.loc[mask_local, "RC"].values

    # curva del modello polinomiale (log–exp) per D fissata
    RC_curve = [np.exp(model.predict([[it, valore_D]])[0]) for it in IT_range]

    col.markdown("### Diagrammi")
    col_fig = col.columns(1)[0]

    # Figura con due pannelli: sinistra 2D, destra 3D
    fig = plt.figure(figsize=(7.6, 2.6), dpi=130)
    fig.subplots_adjust(wspace=0.26)

    # --- Pannello sinistro: modello + dati locali ---
    ax1 = fig.add_subplot(1, 2, 1)
    if len(IT_local) > 0:
        ax1.scatter(
            IT_local, RC_local,
            s=15, alpha=0.7,
            label=f"Dati locali (+/-{int(TOL_LOCAL*100)}% D)"
        )
    ax1.plot(IT_range, RC_curve, label=f"Modello (grado {grado_poly})", linewidth=1.6)
    ax1.axvline(valore_IT, linestyle="--", linewidth=1, label="IT selezionato")
    ax1.set_xlabel("IT [s]")
    ax1.set_ylabel("RC [MPa]")
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=False)

    # --- Pannello destro: superficie 3D RC(IT, D) + punto selezionato ---
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    IT_vals = df["IT"].values
    D_vals = df["D"].values
    RC_vals = df["RC"].values

    # Superficie triangolata sui dati sperimentali
    ax2.plot_trisurf(IT_vals, D_vals, RC_vals, linewidth=0.2, antialiased=True, alpha=0.9)
    ax2.scatter(valore_IT, valore_D, y_pred, color="red", s=40)

    ax2.set_xlabel("IT [s]")
    ax2.set_ylabel("D [kg/m^3]")
    ax2.set_zlabel("RC [MPa]")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    col_fig.image(buf, use_container_width=False)
    col_fig.caption(
        "Figura: a sinistra andamento di RC in funzione di IT per la densità fissata; "
        "a destra superficie RC(IT, D) ricostruita dai dati sperimentali con evidenza del punto selezionato."
    )



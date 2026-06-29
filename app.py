import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─── Configuration de la page ───────────────────────────────────────────────
st.set_page_config(
    page_title="Analyse Fiabilité - Tramway Casablanca",
    page_icon="🚋",
    layout="wide"
)

st.title("🚋 Analyse de Fiabilité — Tramway de Casablanca")
st.markdown("**RATP Dev Casablanca | Système de Traction Citadis 302**")
st.divider()

# ─── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")
sous_systeme = st.sidebar.selectbox(
    "Sous-système",
    ["Onduleur", "Pantographe", "Tachymètre", "Moteur de traction"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Format des dates")
st.sidebar.code("DD/MM/YYYY HH:MM")
st.sidebar.markdown("*Exemple : 15/03/2024 08:30*")

# ─── Saisie des données ─────────────────────────────────────────────────────
st.header(f"📥 Saisie des données — {sous_systeme}")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔴 Dates de début de panne")
    dates_panne_txt = st.text_area(
        "Une date par ligne (DD/MM/YYYY HH:MM) — de la plus récente à la plus ancienne",
        height=300,
        placeholder="20/05/2026 10:31\n31/03/2026 08:00\n16/03/2026 09:00\n...",
        key="pannes"
    )

with col2:
    st.subheader("🟢 Dates de remise en service")
    dates_remise_txt = st.text_area(
        "Une date par ligne (DD/MM/YYYY HH:MM) — de la plus récente à la plus ancienne",
        height=300,
        placeholder="21/05/2026 08:00\n31/03/2026 11:00\n16/03/2026 12:00\n...",
        key="remises"
    )

# ─── Fonctions de calcul ────────────────────────────────────────────────────
def parse_dates(text):
    dates = []
    formats = ["%d/%m/%Y %H:%M", "%d/%m/%Y", "%Y-%m-%d %H:%M", "%Y-%m-%d"]
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parsed = None
        for fmt in formats:
            try:
                parsed = datetime.strptime(line, fmt)
                break
            except:
                continue
        if parsed:
            dates.append(parsed)
        else:
            st.warning(f"⚠️ Date non reconnue : `{line}` — ignorée")
    return dates

def weibull_mle(tbf_hours):
    n = len(tbf_hours)
    data = np.array(tbf_hours)

    def neg_log_likelihood(params):
        beta, eta = params
        if beta <= 0 or eta <= 0:
            return 1e10
        ll = n * np.log(beta) - n * beta * np.log(eta) + \
             (beta - 1) * np.sum(np.log(data)) - \
             np.sum((data / eta) ** beta)
        return -ll

    beta0 = 1.5
    eta0 = np.mean(data)

    result = minimize(neg_log_likelihood, [beta0, eta0],
                      method='Nelder-Mead',
                      options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 10000})
    beta, eta = result.x
    return abs(beta), abs(eta)

def calcul_fiabilite(t, beta, eta):
    R = np.exp(-(t / eta) ** beta)
    F = 1 - R
    f = (beta / eta) * (t / eta) ** (beta - 1) * np.exp(-(t / eta) ** beta)
    lam = (beta / eta) * (t / eta) ** (beta - 1)
    return R, F, f, lam

def interpretation_beta(beta):
    if beta < 1:
        return "🔵 **β < 1** : Mortalité infantile — pannes de jeunesse (défauts de fabrication ou d'installation)"
    elif 0.95 <= beta <= 1.05:
        return "🟢 **β ≈ 1** : Taux de panne constant — pannes aléatoires (loi exponentielle)"
    elif 1 < beta < 3:
        return "🟡 **β entre 1 et 3** : Usure progressive — pannes dues à la fatigue ou dégradation"
    else:
        return "🔴 **β > 3** : Usure prononcée — vieillissement rapide, maintenance préventive urgente"

def recommandation_maintenance(beta, sous_systeme):
    if beta < 1:
        return (
            f"Le sous-système **{sous_systeme}** présente des défaillances précoces (β < 1). "
            "Il est recommandé de renforcer les contrôles à la réception et à l'installation, "
            "et d'effectuer un rodage avant mise en service."
        )
    elif 0.95 <= beta <= 1.05:
        return (
            f"Le sous-système **{sous_systeme}** présente des pannes aléatoires (β ≈ 1). "
            "La maintenance corrective est appropriée. Une surveillance périodique légère suffit."
        )
    elif 1 < beta < 3:
        return (
            f"Le sous-système **{sous_systeme}** montre une usure progressive (β entre 1 et 3). "
            "Une maintenance préventive systématique est recommandée selon le temps d'inspection calculé."
        )
    else:
        return (
            f"Le sous-système **{sous_systeme}** présente une usure sévère (β > 3). "
            "Une maintenance préventive urgente et fréquente est nécessaire, "
            "avec remplacement planifié des composants avant la fin de leur durée de vie."
        )

# ─── Bouton Analyser ────────────────────────────────────────────────────────
if st.button("🔍 Lancer l'analyse", type="primary", use_container_width=True):

    if not dates_panne_txt.strip() or not dates_remise_txt.strip():
        st.error("❌ Veuillez saisir les dates de panne et de remise en service.")
        st.stop()

    pannes_raw = parse_dates(dates_panne_txt)
    remises_raw = parse_dates(dates_remise_txt)

    # Inversion : l'utilisateur saisit du plus récent au plus ancien
    pannes = list(reversed(pannes_raw))
    remises = list(reversed(remises_raw))

    if len(pannes) != len(remises):
        st.error(f"❌ Nombre de dates incohérent : {len(pannes)} pannes vs {len(remises)} remises en service.")
        st.stop()

    if len(pannes) < 3:
        st.error("❌ Minimum 3 pannes nécessaires pour l'analyse Weibull.")
        st.stop()

    # ── Calcul TTR et TBF ──
    ttr_hours = [(r - p).total_seconds() / 3600 for p, r in zip(pannes, remises)]
    tbf_hours = []
    for i in range(1, len(remises)):
        tbf = (pannes[i] - remises[i-1]).total_seconds() / 3600
        if tbf <= 0:
            tbf = 0
        tbf_hours.append(tbf)

    if len(tbf_hours) < 2:
        st.error("❌ Pas assez de TBF calculés.")
        st.stop()

    tbf_non_nuls = [t for t in tbf_hours if t > 0]
    mttr = np.mean(ttr_hours)
    mtbf = np.mean(tbf_non_nuls) if tbf_non_nuls else 0

    # ── Paramètres Weibull ──
    tbf_weibull = tbf_non_nuls
    if len(tbf_weibull) < 2:
        st.error("❌ Pas assez de TBF non nuls pour l'estimation Weibull.")
        st.stop()
    try:
        beta, eta = weibull_mle(tbf_weibull)
    except Exception as e:
        st.error(f"❌ Erreur lors de l'estimation Weibull : {e}")
        st.stop()

    # ── Temps d'inspection (R = 75%) ──
    R_cible = 0.75
    t_insp = eta * (-np.log(R_cible)) ** (1 / beta)
    t_insp_jours = t_insp / 24

    # ── Disponibilité opérationnelle ──
    dispo = mtbf / (mtbf + mttr) * 100 if (mtbf + mttr) > 0 else 0

    # ─── Affichage des résultats ────────────────────────────────────────────
    st.divider()
    st.header(f"📊 Résultats — {sous_systeme}")

    # KPIs
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Nombre de pannes", len(pannes))
    col2.metric("N TBF non nuls", len(tbf_non_nuls))
    col3.metric("MTBF", f"{mtbf:.1f} h")
    col4.metric("MTTR", f"{mttr:.1f} h")
    col5.metric("β (Weibull)", f"{beta:.3f}")
    col6.metric("η (Weibull)", f"{eta:.1f} h")

    st.divider()

    # ── Tableau TBF/TTR ──
    st.subheader("📋 Tableau des TBF et TTR")
    n_rows = min(len(tbf_hours), len(ttr_hours))
    df = pd.DataFrame({
        "N°": range(1, n_rows + 1),
        "Début panne": [pannes[i].strftime("%d/%m/%Y %H:%M") for i in range(n_rows)],
        "Remise en service": [remises[i].strftime("%d/%m/%Y %H:%M") for i in range(n_rows)],
        "TTR (heures)": [round(ttr_hours[i], 2) for i in range(n_rows)],
        "TBF (heures)": [round(tbf_hours[i], 2) if i < len(tbf_hours) else "-" for i in range(n_rows)]
    })
    st.dataframe(df, use_container_width=True)

    st.divider()

    # ── Interprétation Weibull ──
    st.subheader("🔎 Interprétation des paramètres Weibull")
    st.markdown(f"""
    | Paramètre | Valeur | Signification |
    |-----------|--------|---------------|
    | **β (forme)** | {beta:.4f} | Forme de la distribution |
    | **η (échelle)** | {eta:.2f} h | Durée caractéristique (63.2% des pannes) |
    | **MTBF empirique** | {mtbf:.2f} h | Moyenne des TBF non nuls observés |
    | **Disponibilité** | {dispo:.2f} % | MTBF / (MTBF + MTTR) |
    """)

    st.info(interpretation_beta(beta))
    st.success(f"⏱️ **Temps d'inspection recommandé** (R = 75%) : **{t_insp:.1f} heures** soit environ **{t_insp_jours:.1f} jours**")

    st.divider()

    # ── Courbes R(t), F(t), f(t), λ(t) ──
    st.subheader("📈 Courbes de fiabilité")

    t_max = max(tbf_weibull) * 1.5
    t = np.linspace(0.1, t_max, 500)
    R, F, f, lam = calcul_fiabilite(t, beta, eta)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("R(t) — Fiabilité", "F(t) — Probabilité de défaillance",
                        "f(t) — Densité de probabilité", "λ(t) — Taux de défaillance"),
        vertical_spacing=0.15
    )

    fig.add_trace(go.Scatter(x=t, y=R, mode='lines', name='R(t)',
                              line=dict(color='#2ecc71', width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=F, mode='lines', name='F(t)',
                              line=dict(color='#e74c3c', width=2.5)), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=f, mode='lines', name='f(t)',
                              line=dict(color='#3498db', width=2.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=lam, mode='lines', name='λ(t)',
                              line=dict(color='#e67e22', width=2.5)), row=2, col=2)

    for row, col in [(1,1), (1,2), (2,1), (2,2)]:
        fig.add_vline(x=t_insp, line_dash="dash", line_color="purple",
                      annotation_text=f"t_insp={t_insp:.0f}h", row=row, col=col)

    fig.update_layout(height=600, showlegend=False,
                      title_text=f"Courbes de fiabilité Weibull — {sous_systeme} (β={beta:.3f}, η={eta:.1f}h)")
    fig.update_xaxes(title_text="Temps (heures)")
    st.plotly_chart(fig, use_container_width=True)

    # ── Papier de Weibull ──
    st.subheader("📄 Papier de Weibull")

    tbf_sorted = np.sort(tbf_weibull)
    n = len(tbf_sorted)
    F_empirique = (np.arange(1, n+1) - 0.3) / (n + 0.4)
    y_weibull = np.log(-np.log(1 - F_empirique))
    x_weibull = np.log(tbf_sorted)

    slope, intercept, r_value, _, _ = stats.linregress(x_weibull, y_weibull)
    x_line = np.linspace(min(x_weibull), max(x_weibull), 100)
    y_line = slope * x_line + intercept

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x_weibull, y=y_weibull, mode='markers',
                               name='Données empiriques',
                               marker=dict(color='#e74c3c', size=8)))
    fig2.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines',
                               name=f'Droite Weibull (R²={r_value**2:.4f})',
                               line=dict(color='#2c3e50', width=2)))
    fig2.update_layout(
        title=f"Papier de Weibull — {sous_systeme}",
        xaxis_title="ln(TBF)",
        yaxis_title="ln(-ln(1-F))",
        height=450
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(f"R² = {r_value**2:.4f} — {'✅ Bonne adéquation Weibull' if r_value**2 > 0.9 else '⚠️ Adéquation moyenne, vérifier les données'}")

    st.divider()

    # ── Plan de maintenance préventive ──
    st.subheader("🔧 Plan de maintenance préventive")

    st.markdown(recommandation_maintenance(beta, sous_systeme))

    # Génération du calendrier de maintenance
    date_debut = remises[-1] if remises else datetime.now()
    nb_interventions = 6
    interventions = []
    for i in range(1, nb_interventions + 1):
        date_interv = date_debut + timedelta(hours=t_insp * i)
        R_val = np.exp(-(t_insp * i / eta) ** beta)
        interventions.append({
            "Intervention N°": i,
            "Date prévue": date_interv.strftime("%d/%m/%Y"),
            "Délai depuis dernière remise (h)": round(t_insp * i, 1),
            "Fiabilité attendue R(t)": f"{R_val*100:.1f} %",
            "Type": "Maintenance préventive systématique"
        })

    df_maint = pd.DataFrame(interventions)
    st.dataframe(df_maint, use_container_width=True)

    st.info(f"📅 Les interventions sont planifiées tous les **{t_insp:.0f} heures** ({t_insp_jours:.1f} jours) à partir de la dernière remise en service.")

    st.divider()

    # ── Rapport des résultats ──
    st.subheader("📝 Rapport des résultats")

    rapport = f"""
═══════════════════════════════════════════════════════════════
       RAPPORT D'ANALYSE DE FIABILITÉ — TRAMWAY CASABLANCA
              RATP Dev Casablanca | Citadis 302
═══════════════════════════════════════════════════════════════

Sous-système analysé  : {sous_systeme}
Date du rapport       : {datetime.now().strftime("%d/%m/%Y %H:%M")}

───────────────────────────────────────────────────────────────
1. DONNÉES
───────────────────────────────────────────────────────────────
  Nombre total de pannes         : {len(pannes)}
  Nombre de TBF non nuls         : {len(tbf_non_nuls)}
  Nombre de TBF nuls (chevauch.) : {len(tbf_hours) - len(tbf_non_nuls)}

───────────────────────────────────────────────────────────────
2. INDICATEURS DE FIABILITÉ
───────────────────────────────────────────────────────────────
  MTBF (Mean Time Between Failures) : {mtbf:.2f} heures
  MTTR (Mean Time To Repair)        : {mttr:.2f} heures
  Disponibilité opérationnelle      : {dispo:.2f} %

───────────────────────────────────────────────────────────────
3. PARAMÈTRES DE LA LOI DE WEIBULL
───────────────────────────────────────────────────────────────
  β (paramètre de forme)   : {beta:.4f}
  η (paramètre d'échelle)  : {eta:.2f} heures
  R² (adéquation)          : {r_value**2:.4f}

  Interprétation :
  {"Mortalité infantile (β < 1) : défauts de jeunesse" if beta < 1 else "Pannes aléatoires (β ≈ 1)" if abs(beta-1) < 0.05 else "Usure progressive (1 < β < 3)" if beta < 3 else "Usure sévère (β > 3)"}

───────────────────────────────────────────────────────────────
4. RECOMMANDATION DE MAINTENANCE
───────────────────────────────────────────────────────────────
  Taux de fiabilité cible          : 75 %
  Temps d'inspection recommandé    : {t_insp:.1f} heures
                                     ({t_insp_jours:.1f} jours)

  Fréquence des interventions      : Tous les {t_insp:.0f} heures

───────────────────────────────────────────────────────────────
5. PLAN DE MAINTENANCE (6 PROCHAINES INTERVENTIONS)
───────────────────────────────────────────────────────────────
"""
    for interv in interventions:
        rapport += f"  Intervention {interv['Intervention N°']} : {interv['Date prévue']} — R(t) = {interv['Fiabilité attendue R(t)']}\n"

    rapport += """
═══════════════════════════════════════════════════════════════
  Rapport généré automatiquement par l'outil d'analyse PFE
  RATP Dev Casablanca — Système de traction Citadis 302
═══════════════════════════════════════════════════════════════
"""

    st.text(rapport)

    st.divider()

    # ── Export ──
    st.subheader("💾 Export des résultats")
    col1, col2, col3 = st.columns(3)

    with col1:
        csv = df.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')
        st.download_button(
            label="⬇️ Tableau TBF/TTR (CSV)",
            data=csv,
            file_name=f"tbf_ttr_{sous_systeme.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )

    with col2:
        csv_maint = df_maint.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')
        st.download_button(
            label="⬇️ Plan de maintenance (CSV)",
            data=csv_maint,
            file_name=f"maintenance_{sous_systeme.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )

    with col3:
        st.download_button(
            label="⬇️ Rapport complet (TXT)",
            data=rapport.encode('utf-8'),
            file_name=f"rapport_{sous_systeme.lower().replace(' ', '_')}.txt",
            mime="text/plain"
        )

st.divider()
st.caption("Développé dans le cadre du PFE — Analyse de fiabilité du système de traction | RATP Dev Casablanca 2026")
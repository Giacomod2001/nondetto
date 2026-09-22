"""
NonDetto — quello che i clienti non dicono ad alta voce, ma scrivono.

Prototipo. Legge recensioni e ticket, li classifica, e restituisce a ogni
reparto solo la parte che lo riguarda.

Avvio:  streamlit run app.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))
from src import pipeline as pl  # noqa: E402

# ---------------------------------------------------------------------------
# Palette: slot categoriali in ordine fisso + colori di stato riservati.
# I colori di stato non vengono mai riusati per una serie e viaggiano sempre
# con un'etichetta, mai da soli.
# ---------------------------------------------------------------------------
SERIE = ["#2a78d6", "#eb6834", "#1baf7a"]
MESI_IT = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
           "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"]


def mese_it(d):
    return f"{MESI_IT[d.month - 1]} {d.year}"


STATO = {"critico": "#d03b3b", "attenzione": "#fab219", "buono": "#0ca30c"}
INK = "#0b0b0b"
INK_2 = "#52514e"
GRIGLIA = "#e6e5e1"

st.set_page_config(page_title="NonDetto", page_icon="•", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1180px;}
      h1, h2, h3 {letter-spacing: -0.02em;}
      [data-testid="stMetricValue"] {font-size: 1.9rem;}
      .nota {color:#52514e; font-size:0.86rem; line-height:1.45;}
    </style>
    """,
    unsafe_allow_html=True,
)


def stile(fig, altezza=380, titolo=None, legenda=False):
    fig.update_layout(
        height=altezza,
        title=dict(text=titolo, font=dict(size=15, color=INK), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color=INK_2),
        margin=dict(l=8, r=24, t=44 if titolo else 12, b=8),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=GRIGLIA,
                        font=dict(color=INK, size=12)),
        showlegend=legenda,
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRIGLIA, gridwidth=1,
                     zeroline=False, linecolor=GRIGLIA, ticks="")
    fig.update_yaxes(showgrid=False, zeroline=False, linecolor=GRIGLIA, ticks="")
    return fig


@st.cache_data(show_spinner="Analisi in corso…")
def elabora(rec_bytes=None, tck_bytes=None):
    if rec_bytes is not None and tck_bytes is not None:
        import io
        df = pl.carica(io.BytesIO(rec_bytes), io.BytesIO(tck_bytes))
    else:
        df = pl.carica()
    return pl.analizza(df)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### NonDetto")
    st.caption("Prototipo — Giacomo Dellacqua")

    st.markdown("**Dati in ingresso**")
    su_rec = st.file_uploader("Recensioni (CSV)", type="csv")
    su_tck = st.file_uploader("Ticket (CSV)", type="csv")
    if su_rec is None or su_tck is None:
        st.caption("Nessun file caricato: uso il dataset dimostrativo.")

    st.divider()
    st.caption(
        "Le assunzioni del modello economico si regolano dentro la scheda "
        "«Il costo del non detto», accanto ai numeri che cambiano.")

df = elabora(
    su_rec.getvalue() if su_rec else None,
    su_tck.getvalue() if su_tck else None,
)
qualita = pl.accuratezza(df)
costo = pl.costo_del_non_detto(df)
mesi = max(1, round((df["data"].max() - df["data"].min()).days / 30))

# ---------------------------------------------------------------------------
# Intestazione
# ---------------------------------------------------------------------------
st.title("NonDetto")
st.markdown(
    "<p class='nota'>Quello che i clienti non dicono ad alta voce, ma scrivono. "
    "Recensioni e ticket letti insieme, classificati, e consegnati al reparto "
    "che può fare qualcosa.</p>",
    unsafe_allow_html=True,
)
st.warning(
    "**Dati dimostrativi sintetici.** Non provengono da MySecretCase. "
    "Riproducono i temi che emergono pubblicamente dalle recensioni di un "
    "e-commerce del settore. Servono a mostrare come funziona la pipeline."
)

t1, t2, t3, t4, t5, t6 = st.tabs([
    "Quadro generale",
    "Prodotto & Acquisti",
    "Contenuti & Marketing",
    "Operations & Care",
    "Il costo del non detto",
    "Come funziona",
])

# ---------------------------------------------------------------------------
# 1. Quadro generale
# ---------------------------------------------------------------------------
with t1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Conversazioni analizzate", f"{len(df):,}".replace(",", "."))
    c2.metric("Evitabili a monte", f"{costo['quota_sul_totale']:.0%}",
              help="Nate da un'informazione che il cliente non ha trovato.")
    alta = (df["urgenza"] == "alta").sum()
    c3.metric("Da guardare subito", f"{alta}",
              help="Segnali di cliente che ha già scritto senza risposta, "
                   "o che è alla seconda volta.")
    c4.metric("Classificate", f"{qualita['copertura']:.0%}",
              help="Quota di messaggi a cui la pipeline assegna un tema.")

    st.markdown("")
    col_a, col_b = st.columns([1.15, 1])

    with col_a:
        temi = (df[df["intento"] != "feedback_positivo"]["etichetta"]
                .value_counts().sort_values())
        fig = go.Figure(go.Bar(
            x=temi.values, y=temi.index, orientation="h",
            marker=dict(color=SERIE[0], line=dict(width=0)),
            text=temi.values, textposition="outside",
            textfont=dict(color=INK_2, size=11),
            hovertemplate="%{y}<br>%{x} conversazioni<extra></extra>",
        ))
        fig.update_traces(marker_cornerradius=4)
        fig.update_xaxes(range=[0, temi.max() * 1.18])
        st.plotly_chart(stile(fig, 420, "Di cosa scrivono i clienti"),
                        use_container_width=True)

    with col_b:
        # Il rating e' l'unica colonna che dice se un tema fa danno o no.
        # Un tema con tante conversazioni ma stelle alte non e' un problema;
        # uno con poche conversazioni e una stella lo e'.
        rec = df[(df["origine"] == "recensione") & (df["rating"].notna())]
        r = (rec.groupby("etichetta")
             .agg(rating=("rating", "mean"), n=("id", "count"))
             .reset_index())
        r = r[r["n"] >= 10].sort_values("rating", ascending=False)
        media = rec["rating"].mean()

        fig2 = go.Figure(go.Bar(
            x=r["rating"], y=r["etichetta"], orientation="h",
            marker=dict(color=SERIE[0], line=dict(width=0)),
            text=[f"{v:.1f}" for v in r["rating"]], textposition="outside",
            textfont=dict(color=INK_2, size=11), cliponaxis=False,
            customdata=r["n"],
            hovertemplate="%{y}<br>%{x:.2f} stelle su %{customdata} recensioni"
                          "<extra></extra>",
        ))
        fig2.update_traces(marker_cornerradius=4)
        fig2.add_vline(x=media, line=dict(color="#9A8D93", width=2, dash="dash"),
                       annotation_text=f"media {media:.1f}",
                       annotation_position="top",
                       annotation_font=dict(color=INK_2, size=11))
        fig2.update_xaxes(range=[1, 5.6], dtick=1)
        st.plotly_chart(
            stile(fig2, 420, "Quali temi tirano giù le stelle"),
            use_container_width=True)

    st.markdown(
        f"<p class='nota'>{len(df[df.origine=='recensione'])} recensioni e "
        f"{len(df[df.origine=='ticket'])} ticket da {mese_it(df['data'].min())} "
        f"a {mese_it(df['data'].max())}, letti insieme: è la stessa persona che "
        "prima scrive all'assistenza e poi lascia una stella, ma di solito i due "
        "testi li legge gente diversa. &nbsp;·&nbsp; I due grafici vanno letti "
        "in coppia: a sinistra <b>quanto</b> se ne parla, a destra <b>quanto fa "
        "male</b>. Un tema in alto a sinistra e in basso a destra è quello da "
        "cui partire.</p>",
        unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 2. Prodotto & Acquisti
# ---------------------------------------------------------------------------
with t2:
    st.subheader("Cosa si rompe, e su quale prodotto")
    st.markdown(
        "<p class='nota'>Un difetto diffuso su tutto il catalogo è rumore. "
        "Un difetto concentrato su due codici è un problema di fornitore. "
        "L'incidenza è la quota di conversazioni su quel prodotto che parlano "
        "di un guasto.</p>", unsafe_allow_html=True)

    al = pl.alert_prodotto(df)
    top = (al.groupby(["sku", "prodotto"])
           .agg(segnalazioni=("segnalazioni", "sum"),
                incidenza=("incidenza", "sum"))
           .reset_index().sort_values("incidenza"))
    top = top[top["segnalazioni"] >= 3]

    colori = [STATO["critico"] if v >= 0.15
              else STATO["attenzione"] if v >= 0.08
              else SERIE[0] for v in top["incidenza"]]

    fig = go.Figure(go.Bar(
        x=top["incidenza"], y=top["prodotto"], orientation="h",
        marker=dict(color=colori, line=dict(width=0)),
        text=[f"{v:.0%}  ({n})" for v, n in zip(top["incidenza"],
                                                top["segnalazioni"])],
        textposition="outside", textfont=dict(color=INK_2, size=11),
        hovertemplate="%{y}<br>incidenza %{x:.1%}<extra></extra>",
    ))
    fig.update_traces(marker_cornerradius=4)
    fig.update_xaxes(tickformat=".0%", range=[0, top["incidenza"].max() * 1.3])
    st.plotly_chart(stile(fig, 400, "Incidenza delle segnalazioni di guasto"),
                    use_container_width=True)
    st.markdown(
        f"<p class='nota'>🔴 sopra il 15% &nbsp;·&nbsp; 🟡 tra 8% e 15% "
        "&nbsp;·&nbsp; 🔵 sotto l'8%. Le soglie sono un punto di partenza da "
        "tarare insieme a chi gestisce il catalogo.</p>",
        unsafe_allow_html=True)

    st.markdown("**Dettaglio per tipo di guasto**")
    st.dataframe(
        al[["sku", "prodotto", "etichetta", "segnalazioni", "incidenza",
            "rating_medio"]].rename(columns={
                "etichetta": "tipo di guasto", "rating_medio": "rating medio"}),
        use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# 3. Contenuti & Marketing
# ---------------------------------------------------------------------------
with t3:
    st.subheader("Ogni domanda ricorrente è un contenuto che manca")
    st.markdown(
        "<p class='nota'>MySecretCase non può comprare pubblicità come chiunque "
        "altro: il traffico se lo guadagna con i contenuti. Quindi sapere quali "
        "domande le persone fanno davvero, e su quale prodotto, non è un "
        "dettaglio operativo — è il piano editoriale.</p>",
        unsafe_allow_html=True)

    backlog = pl.backlog_contenuti(df, top=18)
    c1, c2 = st.columns([1, 1.3])
    with c1:
        per_tema = df[df["evitabile"]]["etichetta"].value_counts().sort_values()
        fig = go.Figure(go.Bar(
            x=per_tema.values, y=per_tema.index, orientation="h",
            marker=dict(color=SERIE[0], line=dict(width=0)),
            text=per_tema.values, textposition="outside",
            textfont=dict(color=INK_2, size=11),
            hovertemplate="%{y}<br>%{x} richieste<extra></extra>",
        ))
        fig.update_traces(marker_cornerradius=4)
        fig.update_xaxes(range=[0, per_tema.max() * 1.22])
        st.plotly_chart(stile(fig, 260, "Domande che nascono da un vuoto informativo"),
                        use_container_width=True)
    with c2:
        st.markdown("**Backlog editoriale, già ordinato per urgenza**")
        st.dataframe(
            backlog.drop(columns="categoria").rename(
                columns={"dove_intervenire": "dove intervenire"}),
            use_container_width=True, hide_index=True, height=260,
            column_config={
                "tema": st.column_config.TextColumn(width="small"),
                "richieste": st.column_config.NumberColumn(width="small"),
                "dove intervenire": st.column_config.TextColumn(width="medium"),
            })

    st.markdown("**Le domande, testuali**")
    tema_scelto = st.selectbox(
        "Tema", sorted(df[df["evitabile"]]["etichetta"].unique()))
    esempi = df[(df["evitabile"]) & (df["etichetta"] == tema_scelto)]
    for t in esempi["testo"].head(8):
        st.markdown(f"> {t}")

# ---------------------------------------------------------------------------
# 4. Operations & Care
# ---------------------------------------------------------------------------
with t4:
    st.subheader("Dove finisce il tempo dell'assistenza")

    ops = df[df["reparto"] == "Operations & Care"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Conversazioni operative", len(ops))
    c2.metric("Urgenza alta", int((ops["urgenza"] == "alta").sum()))
    ripetute = ops["etichetta"].value_counts()
    c3.metric("Tema più pesante", ripetute.index[0], f"{ripetute.iloc[0]} casi")

    piv = (df[df["reparto"] != "—"]
           .pivot_table(index="etichetta", columns="canale",
                        values="id", aggfunc="count").fillna(0))
    piv["tot"] = piv.sum(axis=1)
    piv = piv.sort_values("tot", ascending=False).drop(columns="tot")
    st.markdown("**Volumi per tema e canale**")
    st.dataframe(piv.astype(int), use_container_width=True)

    st.markdown("**Coda urgente: chi ha già scritto e non ha avuto risposta**")
    coda = (df[df["urgenza"] == "alta"]
            [["id", "data", "canale", "etichetta", "prodotto", "testo"]]
            .sort_values("data", ascending=False).head(25))
    coda["data"] = coda["data"].dt.strftime("%d/%m/%Y")
    st.dataframe(coda, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# 5. Il costo del non detto
# ---------------------------------------------------------------------------
with t5:
    st.subheader("Il costo del non detto")
    st.markdown(
        "<p class='nota'>Una parte delle conversazioni non doveva esistere: "
        "nasce da un'informazione che il cliente cercava e non ha trovato. "
        "Quella parte si può misurare, e si può ridurre scrivendo, non "
        "assumendo.</p>", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("**Le assunzioni.** Non sono misure: muovile e guarda cosa cambia.")
    a1, a2, a3, a4, a5 = st.columns(5)
    minuti = a1.slider("Minuti per conversazione", 3, 25, 9)
    costo_h = a2.slider("Costo orario care (€)", 12, 40, 22)
    aov = a3.slider("Valore medio ordine (€)", 20, 120, 52)
    perdita = a4.slider("Pre-acquisto che non converte", 0.0, 0.6, 0.25, 0.05)
    volume_reale = a5.number_input("Conversazioni reali/mese", 0, 100000, 0, 500,
                                   help="0 = mostra solo i numeri del campione")

    costo_t5 = pl.costo_del_non_detto(df, minuti, costo_h, aov, perdita)
    per_tema = pl.costo_per_tema(df, minuti, costo_h, aov, perdita)

    # se conosco il volume vero, porto tutto a base annua
    if volume_reale > 0:
        fattore = volume_reale * 12 / (len(df) / mesi * 12)
        unita = "all'anno, sul volume che hai indicato"
    else:
        fattore = 1.0
        unita = f"sul campione di {len(df)} conversazioni (≈{mesi} mesi)"

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Conversazioni evitabili", costo_t5["conversazioni_evitabili"],
              f"{costo_t5['quota_sul_totale']:.0%} del totale", delta_color="off")
    c2.metric("Ore di assistenza",
              f"{costo_t5['ore_care'] * fattore:,.0f} h".replace(",", "."),
              unita, delta_color="off")
    c3.metric("Impatto economico",
              f"{(costo_t5['costo_care_euro'] + costo_t5['mancato_fatturato_euro']) * fattore:,.0f} €".replace(",", "."),
              "assistenza + ordini a rischio", delta_color="off")

    st.markdown("")
    y = per_tema["etichetta"]
    assist = per_tema["costo_assistenza"] * fattore
    ordini = per_tema["ordini_a_rischio"] * fattore
    totali = assist + ordini

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=assist, y=y, orientation="h", name="Costo di assistenza",
        marker=dict(color=SERIE[0], line=dict(color="#ffffff", width=1)),
        hovertemplate="%{y}<br>assistenza: %{x:,.0f} €<extra></extra>"))
    fig.add_trace(go.Bar(
        x=ordini, y=y, orientation="h", name="Ordini a rischio",
        marker=dict(color=SERIE[1], line=dict(color="#ffffff", width=1)),
        text=[f"{t:,.0f} €".replace(",", ".") for t in totali],
        textposition="outside", textfont=dict(color=INK_2, size=12),
        cliponaxis=False,
        hovertemplate="%{y}<br>ordini a rischio: %{x:,.0f} €<extra></extra>"))
    fig.update_layout(barmode="stack",
                      legend=dict(orientation="h", y=-0.22, x=0,
                                  font=dict(color=INK_2, size=11)))
    fig.update_xaxes(range=[0, max(totali.max() * 1.25, 1)], tickformat=",.0f",
                     ticksuffix=" €")
    st.plotly_chart(
        stile(fig, 320, f"Dove sta il costo, {unita}", legenda=True),
        use_container_width=True)

    st.markdown(
        "<p class='nota'>Guarda le due barre in alto. «Quale prodotto fa per me» "
        "ha metà delle conversazioni di «Come si usa», ma pesa di più: perché "
        "una domanda prima dell'acquisto non costa solo tempo di assistenza, "
        "costa un ordine che non si chiude. È questo che decide da dove "
        "cominciare, non il conteggio.</p>", unsafe_allow_html=True)

    if volume_reale > 0:
        st.markdown(
            "<p class='nota'>La proiezione moltiplica il tasso osservato sul "
            "campione per il volume che hai indicato. Vale quanto vale "
            "l'assunzione: serve a dare un ordine di grandezza, non una cifra "
            "da mettere a budget.</p>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Da dove si comincia**")
    prime = pl.backlog_contenuti(df, top=5)
    for _, r in prime.iterrows():
        st.markdown(
            f"- **{r['prodotto']}** — {r['tema'].lower()}: {r['richieste']} "
            f"richieste. → {r['dove_intervenire']}")
    st.markdown(
        "<p class='nota'>Cinque interventi, non cinquanta. Si parte da questi, "
        "si misura se il volume di quelle domande scende, e poi si passa ai "
        "successivi.</p>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 6. Come funziona
# ---------------------------------------------------------------------------
with t6:
    st.subheader("Come funziona, e dove sbaglia")

    st.markdown("""
**La pipeline, in quattro passaggi**

1. **Unione.** Recensioni e ticket entrano nella stessa tabella. Oggi in quasi
   tutte le aziende li leggono due team diversi che non si parlano.
2. **Classificazione.** Ogni messaggio riceve un tema, un reparto e un livello
   di urgenza. Non uso un LLM: uso un lessico scritto in chiaro, che chiunque
   in azienda può aprire e correggere senza saper programmare.
3. **Clustering non supervisionato.** In parallelo, TF-IDF + riduzione LSA +
   KMeans fanno emergere i raggruppamenti che il lessico *non* prevede. Serve a
   non restare ciechi su un problema nuovo.
4. **Consegna.** Ogni reparto vede solo la sua parte, già ordinata.
""")

    st.markdown("**Perché regole e non un modello linguistico**")
    st.markdown("""
- Gira in locale: nessuna conversazione intima esce dall'azienda.
- Costo zero per messaggio: si può far girare su tutto lo storico.
- Verificabile: davanti a una classificazione sbagliata si vede *quale riga*
  l'ha prodotta e si corregge in trenta secondi.
- È la **baseline**: senza un numero di partenza non si può dimostrare che un
  modello più costoso valga la spesa.

Il punto in cui un LLM serve davvero è dare un nome leggibile ai cluster e
generare la bozza del contenuto mancante. Nel codice quel punto è una funzione
sola (`etichetta_cluster_con_llm`), isolata apposta.
""")

    st.markdown("---")
    st.markdown("**Quanto è affidabile**")
    q1, q2, q3 = st.columns(3)
    q1.metric("Accuratezza sul tema", f"{qualita['accuratezza']:.1%}")
    q2.metric("Copertura", f"{qualita['copertura']:.1%}")
    q3.metric("Messaggi valutati", qualita["totale"])

    st.error(
        "**Questo numero è più debole di quanto sembri.** Il dataset è sintetico: "
        "l'ho generato io da modelli di frase, quindi l'accuratezza misura la "
        "coerenza interna della pipeline, non le sue prestazioni su testi veri. "
        "Su dati reali va rifatta su un campione annotato a mano — indicativamente "
        "300 messaggi, mezza giornata di lavoro. Mi aspetto che il numero scenda."
    )

    st.markdown("**Cosa questo prototipo non fa**")
    st.markdown("""
- Non gestisce ironia, sarcasmo o dialetto.
- Non distingue due problemi diversi dentro lo stesso messaggio: ne assegna uno.
- Il modello economico si regge su assunzioni mie, non su dati aziendali.
- Il clustering si tara a occhio sul numero di gruppi: è il pezzo più fragile.
""")

    st.markdown("---")
    st.markdown("**Gruppi emersi dal clustering**")
    cl = (df.groupby("cluster")
          .agg(messaggi=("id", "count"),
               termini=("cluster_termini", "first"),
               tema_prevalente=("etichetta",
                                lambda s: s.value_counts().index[0]))
          .reset_index().sort_values("messaggi", ascending=False))
    st.dataframe(cl, use_container_width=True, hide_index=True)
    st.markdown(
        "<p class='nota'>Quando un gruppo del clustering non coincide con "
        "nessun tema previsto, è lì che c'è qualcosa da guardare a mano.</p>",
        unsafe_allow_html=True)

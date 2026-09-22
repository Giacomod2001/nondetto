"""
NonDetto - pipeline di analisi.

Prende recensioni e ticket in ingresso e produce tre cose:

1. una CLASSIFICAZIONE di ogni messaggio in un'intenzione (perche' il cliente
   ha scritto), fatta con un lessico trasparente e ispezionabile;
2. un CLUSTERING non supervisionato (TF-IDF + KMeans) che fa emergere i temi
   senza deciderli a priori, per non farsi trovare impreparati su qualcosa
   che il lessico non prevede;
3. le AGGREGAZIONI che servono ai tre reparti destinatari.

Scelta progettuale: la classificazione e' basata su regole e non su un LLM.
Motivi: gira in locale senza costi e senza mandare fuori conversazioni
sensibili, e' verificabile riga per riga, ed e' la baseline con cui misurare
qualunque modello piu' complesso si voglia usare dopo.
Il passaggio a un LLM per l'etichettatura dei cluster e' previsto e isolato
in una funzione sola (vedi `etichetta_cluster_con_llm`).
"""

from pathlib import Path
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"

# ---------------------------------------------------------------------------
# 1. Tassonomia: intenzione -> reparto che deve agire
# ---------------------------------------------------------------------------
REPARTO = {
    "difetto_batteria": "Prodotto & Acquisti",
    "difetto_prodotto": "Prodotto & Acquisti",
    "uso_prodotto": "Contenuti & Marketing",
    "scelta_preacquisto": "Contenuti & Marketing",
    "informazione_scheda": "Contenuti & Marketing",
    "logistica_consegna": "Operations & Care",
    "reso_rimborso": "Operations & Care",
    "promo_ordini": "Operations & Care",
    "assistenza": "Operations & Care",
    "imballo_privacy": "Operations & Care",
    "corsi_contenuti": "Contenuti & Marketing",
    "feedback_positivo": "—",
}

ETICHETTA = {
    "difetto_batteria": "Batteria e ricarica",
    "difetto_prodotto": "Difetto o qualità prodotto",
    "uso_prodotto": "Come si usa",
    "scelta_preacquisto": "Quale prodotto fa per me",
    "informazione_scheda": "Scheda prodotto incompleta",
    "logistica_consegna": "Consegna e tracking",
    "reso_rimborso": "Reso e rimborso",
    "promo_ordini": "Promozioni e ordini",
    "assistenza": "Qualità dell'assistenza",
    "imballo_privacy": "Imballo e discrezione",
    "corsi_contenuti": "Corsi e contenuti",
    "feedback_positivo": "Feedback positivo",
}

# Conversazioni che nascono da un'informazione che il cliente non ha trovato.
# Sono quelle evitabili a monte: e' su queste che si misura il "costo del non detto".
EVITABILI = ["uso_prodotto", "scelta_preacquisto", "informazione_scheda"]

# ---------------------------------------------------------------------------
# 2. Lessico. Ordine = priorità: la prima intenzione che matcha vince.
#    Tenuto volutamente leggibile: chiunque in azienda può correggerlo.
# ---------------------------------------------------------------------------
# Ordine = priorita'. A parita' di riscontri vince l'intenzione piu' in alto.
# L'ordine non e' casuale: le domande ("come si carica?") stanno sopra i guasti
# ("non si carica"), altrimenti una richiesta di aiuto finisce fra i difetti.
LESSICO = [
    ("corsi_contenuti", [
        r"\bcors[oi]\b", r"\bvideo corso\b", r"\bvostri articol\w*",
        r"\bpiattaforma\b", r"\btono che usate\b", r"\bguida ai material\w*",
        r"\bcontenuto che spieghi\b", r"\bfare i moralisti\b",
    ]),
    ("feedback_positivo", [
        r"\bconsigliatissim\w*", r"\bottim\w*", r"\bimpeccabil\w*",
        r"\bsoddisfatt\w*", r"\bperfett\w*", r"\bsuperato le aspettative\b",
        r"\bvale ogni euro\b", r"\bqualit\u00e0 alta\b", r"\bcomprer\u00f2 ancora\b",
        r"\btutto come promesso\b", r"\bcomplimenti\b", r"\bvelocissim\w*",
        r"\bben rifinit\w*", r"\besattamente come lo descrivete\b",
        r"\bnegozio italiano serio\b", r"\btornata per la seconda volta\b",
    ]),
    ("scelta_preacquisto", [
        r"\bconsigli\w*", r"\bprimo acquisto\b", r"\bprime armi\b",
        r"\bche taglia\b", r"\bregalo\b", r"\bindecis\w*", r"\bdifferenza tra\b",
        r"\bmi oriento\b", r"\badatt[oa]\b", r"\bsto scegliendo\b",
        r"\bda dove iniziare\b", r"\bsilenzios\w*", r"\bpelle sensibile\b",
        r"\bgravidanza\b", r"\bpavimento pelvico\b", r"\bguida che li confronti\b",
        r"\bsenza app\b", r"\bpaura di sbagliare\b", r"\btroppo intenso\b",
        r"\bin coppia a distanza\b", r"\bstessa categoria\b",
    ]),
    ("uso_prodotto", [
        r"\bcome si (carica|usa|pulisce|collega|conserva|fa)\b",
        r"\bcome faccio a capire\b", r"\bmodo corretto di\b",
        r"\bistruzion\w*", r"\blibrett\w*", r"\bfoglietto\b",
        r"\bnon riesco a colleg\w*", r"\bbluetooth\b", r"\bsi disconnette\b",
        r"\bpulire\b", r"\bdetergente\b", r"\bmodalit\u00e0\b",
        r"\bblocco tasti\b", r"\bsotto la doccia\b", r"\bsi pu\u00f2 usare\b",
        r"\blubrificante al silicone\b", r"\bdove si conserva\b",
        r"\bnon capisco a cosa servano\b", r"\bcustodia\b",
    ]),
    ("informazione_scheda", [
        r"\bsched\w*", r"\bdescrizion\w*", r"\bdimension\w*", r"\bmisur\w*",
        r"\bdecibel\b", r"\ble foto\b", r"\bnon sono indicat\w*",
        r"\bnon \u00e8 (chiaro|scritta|indicat\w*)\b",
        r"\binformazioni sul prodotto\b", r"\binsufficient\w*",
        r"\bmanca l.indicazione\b", r"\bcosa sia incluso\b", r"\balla cieca\b",
        r"\bnon si capisce dalla descrizione\b", r"\bmolto vaga\b",
    ]),
    ("reso_rimborso", [
        r"\bres[oi]\b", r"\brimbors\w*", r"\brestitu\w*", r"\bsostitu\w*",
        r"\bil cambio\b", r"\bprodotto diverso\b",
    ]),
    ("difetto_batteria", [
        r"\bbatteri\w*", r"\bnon tiene la carica\b", r"\bsi scarica\b",
        r"\bgi\u00e0 scarico\b", r"\bnon si accende\b", r"\bsi spegne da solo\b",
        r"\bbase magnetica\b", r"\bricarica magnetica\b",
        r"\bcavo di ricarica\b", r"\bimpossibile lasciarlo in carica\b",
    ]),
    ("difetto_prodotto", [
        r"\bdifett\w*", r"\bsi \u00e8 rott\w*", r"\bsi \u00e8 strappat\w*",
        r"\bun taglio\b", r"\bodore\b", r"\bpi\u00f9 rumoroso\b",
        r"\bsi sente da fuori\b", r"\bnon risponde\b", r"\bgi\u00e0 usato\b",
        r"\bsmesso di vibrare\b", r"\bimpermeabil\w*",
        r"\bfare le bizze\b", r"\bconfezione era chiaramente\b",
    ]),
    ("logistica_consegna", [
        r"\bspedizion\w*", r"\bconsegn\w*", r"\bcorrier\w*", r"\btracking\b",
        r"\bpacco non\b", r"\bnon \u00e8 ancora arrivat\w*", r"\bin ritardo\b",
        r"\bin transito\b", r"\bexpress\b", r"\bdue spedizioni\b",
        r"\bordine diviso\b", r"\bal vicino\b", r"\b48 ore, ne sono passate\b",
    ]),
    ("imballo_privacy", [
        r"\bimball\w*", r"\bdiscret\w*", r"\bdiscrezion\w*", r"\banonim\w*",
        r"\bdoppia scatola\b", r"\bestratto conto\b", r"\bla bolla\b",
        r"\bpackaging\b", r"\bconfezione curata\b", r"\bnome che compare\b",
        r"\bsulla carta \u00e8 riconoscibile\b", r"\bsulla scatola\b",
        r"\bvivo con i miei\b", r"\bvivo in condivisione\b",
    ]),
    ("promo_ordini", [
        r"\bcodice sconto\b", r"\bpromozion\w*", r"\bfattur\w*",
        r"\bbonus\b", r"\bnewsletter\b", r"\bprezzo pieno\b",
        r"\bspedizione gratuita\b", r"\barea personale\b",
        r"\bmail con gli ordini\b", r"\bspam\b",
    ]),
    ("assistenza", [
        r"\bassistenz\w*", r"\bnessuna rispost\w*", r"\brisposte vagh\w*",
        r"\bcopiate e incollate\b", r"\btre persone diverse\b",
        r"\brisposta chiara\b", r"\bsenza giudizio\b",
        r"\bnon ho ricevuto nessuna\b", r"\bmezza giornata\b",
        r"\bnon risolutiva\b", r"\bdopo cinque giorni\b",
    ]),
]

# Se compare uno di questi segnali, il messaggio NON e' un semplice complimento
# anche quando contiene parole positive ("assistenza gentile ma lentissima").
VETO_POSITIVO = [
    r"\bma\b", r"\bper\u00f2\b", r"\bnon\b", r"\bmai\b", r"\britard\w*",
    r"\brott\w*", r"\bdifett\w*", r"\blent\w*", r"\bvagh\w*",
    r"\bscadut\w*", r"\bproblema\b", r"\baspettando\b",
]

LESSICO_COMPILATO = [
    (intento, [re.compile(p, re.IGNORECASE) for p in pattern])
    for intento, pattern in LESSICO
]
VETO_COMPILATO = [re.compile(p, re.IGNORECASE) for p in VETO_POSITIVO]


def classifica(testo: str) -> str:
    """Assegna un'intenzione contando quante espressioni del lessico matchano.
    Vince l'intenzione con piu' riscontri; a parita' vince la piu' prioritaria."""
    ha_veto = any(p.search(testo) for p in VETO_COMPILATO)
    punteggi = []
    for intento, pattern in LESSICO_COMPILATO:
        n = sum(1 for p in pattern if p.search(testo))
        if intento == "feedback_positivo" and ha_veto:
            n = 0  # parole positive dentro una lamentela non fanno un complimento
        punteggi.append((n, intento))
    migliore = max(punteggi, key=lambda x: x[0])
    if migliore[0] == 0:
        return "non_classificato"
    massimo = migliore[0]
    for n, intento in punteggi:  # ordine del lessico = priorità
        if n == massimo:
            return intento
    return "non_classificato"


# ---------------------------------------------------------------------------
# 3. Urgenza: quanto in fretta va guardata questa conversazione
# ---------------------------------------------------------------------------
SEGNALI_URGENZA = [
    r"\bmai arrivat\w*", r"\bnon è ancora arrivat\w*", r"\bnon ho ricevuto\b",
    r"\bseconda volta\b", r"\bterza volta\b", r"\bnessuna rispost\w*",
    r"\bsilenzio totale\b", r"\bpromesso\b", r"\btre settimane\b",
    r"\bventi giorni\b", r"\bgià usato\b", r"\bbatteria era gonfia\b",
]
SEGNALI_URGENZA = [re.compile(p, re.IGNORECASE) for p in SEGNALI_URGENZA]


def urgenza(testo: str, rating=None) -> str:
    colpi = sum(1 for p in SEGNALI_URGENZA if p.search(testo))
    if colpi >= 2 or (colpi >= 1 and rating is not None and rating <= 2):
        return "alta"
    if colpi >= 1:
        return "media"
    return "bassa"


# ---------------------------------------------------------------------------
# 4. Clustering non supervisionato: cosa c'e' di cui non sapevamo nulla
# ---------------------------------------------------------------------------
STOPWORD_IT = """
a ad agli ai al all alla alle allo anche avere aveva avevo c che chi ci cio cioe co coi
col come con cosa cosi cui da dagli dai dal dall dalla dalle dallo degli dei del dell
della delle dello di due e ed egli ecco era erano essere essi fa fare fatto fin fino fu
gia gli grande ha hai hanno ho i il in io la le lei li lo loro lui ma me mi mia mie mio
molto ne negli nei nel nell nella nelle nello no noi non nostra nostro o ogni oppure per
piu po poi perche potere prima qua qual quale quando quanto quel quella quelle quello
questa queste questi questo qui sara sarebbe se sei senza si sia siamo sono sta stata
stati stato su sua sue sui sul sull sulla sulle suo sono ti tra troppo tu tua tue tuo
tutti tutto un una uno vi voi vostro e' po' ho c'e' n'e' me lo la mi ti gli
buongiorno buonasera salve ciao grazie anticipo mille attesa resto favore sapere
aiutarmi potete cordiali saluti prodotto prodotti
""".split()


def normalizza_per_cluster(testo, nomi_prodotto):
    """Toglie i nomi dei prodotti prima di clusterizzare.

    Senza questo passaggio i cluster si formano sul PRODOTTO ("tutti i messaggi
    che citano il Duo App") invece che sul PROBLEMA ("la ricarica non aggancia"),
    e il risultato diventa inutile: il prodotto lo sappiamo gia' dall'ordine,
    quello che non sappiamo e' cosa non funziona.
    """
    t = testo.lower()
    for nome in nomi_prodotto:
        t = t.replace(nome.lower(), " prodotto ")
    return t


def clusterizza(testi, k=12, seed=7):
    """TF-IDF + KMeans. Restituisce l'etichetta di cluster per ogni testo
    e i termini che caratterizzano ciascun cluster."""
    vec = TfidfVectorizer(
        lowercase=True,
        stop_words=STOPWORD_IT,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.4,
        sublinear_tf=True,
    )
    X = vec.fit_transform(testi)

    # LSA prima di KMeans: sulla matrice TF-IDF grezza (migliaia di colonne quasi
    # tutte a zero) le distanze si appiattiscono e KMeans butta tutto in un unico
    # cluster gigante. Riducendo a poche decine di dimensioni i temi si separano.
    lsa = make_pipeline(TruncatedSVD(n_components=60, random_state=seed),
                        Normalizer(copy=False))
    Xr = lsa.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=seed, n_init=20)
    etichette = km.fit_predict(Xr)

    # riporto i centroidi nello spazio delle parole per poterli leggere
    termini = vec.get_feature_names_out()
    centri = lsa[0].inverse_transform(km.cluster_centers_)
    descrizioni = {}
    for i in range(k):
        top = centri[i].argsort()[::-1][:6]
        descrizioni[i] = ", ".join(termini[j] for j in top)
    return etichette, descrizioni


def etichetta_cluster_con_llm(descrizione_termini, esempi):
    """PUNTO DI ESTENSIONE.

    In produzione qui si chiama un LLM per dare al cluster un nome leggibile
    ('Ricarica magnetica che non aggancia') a partire dai termini e da 5 esempi.
    Nel prototipo restituiamo i termini grezzi, cosi' la demo gira offline
    e senza costi. La firma non cambia quando si collega il modello.
    """
    return descrizione_termini


# ---------------------------------------------------------------------------
# 5. Orchestrazione
# ---------------------------------------------------------------------------
def carica(percorso_recensioni=None, percorso_ticket=None) -> pd.DataFrame:
    rec = pd.read_csv(percorso_recensioni or DATA / "recensioni.csv")
    tck = pd.read_csv(percorso_ticket or DATA / "ticket.csv")

    rec["origine"] = "recensione"
    tck["origine"] = "ticket"
    tck["rating"] = pd.NA
    rec["canale"] = rec["fonte"]

    colonne = ["id", "data", "origine", "canale", "sku", "prodotto",
               "categoria", "rating", "testo", "tema_reale"]
    df = pd.concat([rec[colonne], tck[colonne]], ignore_index=True)
    df["data"] = pd.to_datetime(df["data"])
    return df


def analizza(df: pd.DataFrame, k_cluster=12) -> pd.DataFrame:
    df = df.copy()
    df["intento"] = df["testo"].apply(classifica)
    df["etichetta"] = df["intento"].map(ETICHETTA).fillna("Non classificato")
    df["reparto"] = df["intento"].map(REPARTO).fillna("Da rivedere")
    df["urgenza"] = df.apply(
        lambda r: urgenza(r["testo"], r["rating"] if pd.notna(r["rating"]) else None),
        axis=1,
    )
    df["evitabile"] = df["intento"].isin(EVITABILI)

    nomi = sorted(df["prodotto"].dropna().unique(), key=len, reverse=True)
    testi_puliti = [normalizza_per_cluster(t, nomi) for t in df["testo"]]
    etichette, descrizioni = clusterizza(testi_puliti, k=k_cluster)
    df["cluster"] = etichette
    df["cluster_termini"] = df["cluster"].map(descrizioni)
    return df


def accuratezza(df: pd.DataFrame) -> dict:
    """Confronto con il tema usato per generare il testo.

    NOTA ONESTA: il dataset e' sintetico, quindi questo numero misura la
    coerenza interna della pipeline, non le sue prestazioni su dati reali.
    Su dati veri va rifatto su un campione annotato a mano.
    """
    noti = df[df["tema_reale"].notna()]
    corrette = (noti["intento"] == noti["tema_reale"]).sum()
    non_classificati = (df["intento"] == "non_classificato").sum()
    return {
        "totale": len(noti),
        "corrette": int(corrette),
        "accuratezza": round(corrette / len(noti), 3) if len(noti) else 0,
        "non_classificati": int(non_classificati),
        "copertura": round(1 - non_classificati / len(df), 3),
    }


def riepilogo_reparti(df: pd.DataFrame) -> pd.DataFrame:
    g = (df[df["intento"] != "feedback_positivo"]
         .groupby(["reparto", "etichetta"])
         .agg(conversazioni=("id", "count"),
              urgenza_alta=("urgenza", lambda s: (s == "alta").sum()),
              rating_medio=("rating", "mean"))
         .reset_index()
         .sort_values("conversazioni", ascending=False))
    g["rating_medio"] = g["rating_medio"].round(2)
    return g


def costo_del_non_detto(df, minuti_per_conversazione=9, costo_orario=22,
                        valore_ordine=52, tasso_conversione_perso=0.25):
    """Stima l'impatto delle conversazioni nate da informazione mancante.

    Tutti i parametri sono ASSUNZIONI, non misure: vanno sostituiti con i
    dati reali dell'azienda. Sono esposti come parametri proprio per questo.
    """
    evitabili = df[df["evitabile"]]
    n = len(evitabili)
    ore = n * minuti_per_conversazione / 60
    costo_care = ore * costo_orario

    # le domande pre-acquisto sono quelle che possono far perdere un ordine
    preacquisto = (evitabili["intento"] == "scelta_preacquisto").sum()
    mancato_fatturato = preacquisto * tasso_conversione_perso * valore_ordine

    return {
        "conversazioni_evitabili": n,
        "quota_sul_totale": round(n / len(df), 3),
        "ore_care": round(ore, 1),
        "costo_care_euro": round(costo_care),
        "domande_preacquisto": int(preacquisto),
        "mancato_fatturato_euro": round(mancato_fatturato),
    }


def costo_per_tema(df, minuti_per_conversazione=9, costo_orario=22,
                   valore_ordine=52, tasso_conversione_perso=0.25):
    """Scompone il costo del non detto per tema.

    Serve a rispondere alla domanda "da dove comincio": due temi con lo stesso
    numero di conversazioni possono pesare in modo molto diverso, perche' le
    domande pre-acquisto non costano solo tempo, costano ordini.
    """
    ev = df[df["evitabile"]]
    g = (ev.groupby(["intento", "etichetta"])
         .agg(conversazioni=("id", "count")).reset_index())
    g["costo_assistenza"] = g["conversazioni"] * minuti_per_conversazione / 60 * costo_orario
    g["ordini_a_rischio"] = 0.0
    pre = g["intento"] == "scelta_preacquisto"
    g.loc[pre, "ordini_a_rischio"] = (
        g.loc[pre, "conversazioni"] * tasso_conversione_perso * valore_ordine)
    g["totale"] = g["costo_assistenza"] + g["ordini_a_rischio"]
    return g.sort_values("totale")


def backlog_contenuti(df: pd.DataFrame, top=15) -> pd.DataFrame:
    """Ogni domanda ricorrente = un pezzo di contenuto che manca.
    Output pensato per essere passato direttamente a chi scrive."""
    dom = df[df["evitabile"]].copy()
    g = (dom.groupby(["intento", "prodotto", "categoria"])
         .agg(richieste=("id", "count"))
         .reset_index()
         .sort_values("richieste", ascending=False)
         .head(top))
    g["dove_intervenire"] = g["intento"].map({
        "uso_prodotto": "Scheda prodotto — sezione «Come si usa» + FAQ",
        "scelta_preacquisto": "Magazine — guida alla scelta + quiz di orientamento",
        "informazione_scheda": "Scheda prodotto — specifiche tecniche mancanti",
    })
    g["intento"] = g["intento"].map(ETICHETTA)
    g = g.rename(columns={"intento": "tema"})
    return g[["tema", "prodotto", "richieste", "dove_intervenire", "categoria"]]


def alert_prodotto(df: pd.DataFrame) -> pd.DataFrame:
    """Difetti concentrati su SKU specifici: il segnale che deve arrivare
    a chi gestisce catalogo e fornitori."""
    dif = df[df["intento"].isin(["difetto_batteria", "difetto_prodotto"])]
    tot = df.groupby("sku")["id"].count().rename("conversazioni_totali")
    g = (dif.groupby(["sku", "prodotto", "etichetta"])
         .agg(segnalazioni=("id", "count"),
              rating_medio=("rating", "mean"))
         .reset_index()
         .merge(tot, on="sku"))
    g["incidenza"] = (g["segnalazioni"] / g["conversazioni_totali"]).round(3)
    g["rating_medio"] = g["rating_medio"].round(2)
    return g.sort_values("segnalazioni", ascending=False)


if __name__ == "__main__":
    df = analizza(carica())
    print("\n=== Qualità della classificazione ===")
    for k, v in accuratezza(df).items():
        print(f"  {k}: {v}")

    print("\n=== Conversazioni per reparto ===")
    print(df[df["intento"] != "feedback_positivo"]["reparto"].value_counts().to_string())

    print("\n=== Top temi ===")
    print(df["etichetta"].value_counts().head(12).to_string())

    print("\n=== Costo del non detto (con assunzioni di default) ===")
    for k, v in costo_del_non_detto(df).items():
        print(f"  {k}: {v}")

    print("\n=== Alert prodotto ===")
    print(alert_prodotto(df).head(8).to_string(index=False))

    print("\n=== Backlog contenuti ===")
    print(backlog_contenuti(df).head(10).to_string(index=False))

    print("\n=== Cluster emersi ===")
    for c, t in sorted(df.groupby("cluster")["cluster_termini"].first().items()):
        n = (df["cluster"] == c).sum()
        print(f"  cluster {c} ({n} msg): {t}")

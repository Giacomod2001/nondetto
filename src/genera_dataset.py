"""
NonDetto - generatore del dataset dimostrativo.

ATTENZIONE: i dati prodotti da questo script sono SINTETICI.
Non provengono da MySecretCase. Sono costruiti per riprodurre in modo
realistico i temi che emergono pubblicamente dalle recensioni di un
e-commerce di intimita' e benessere sessuale (consegne, resi, batteria,
istruzioni, scelta del prodotto, discrezione del pacco).
Servono unicamente a far girare e a mostrare la pipeline.
"""

import random
import csv
from datetime import date, timedelta
from pathlib import Path

random.seed(7)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OGGI = date(2026, 9, 22)
GIORNI = 180  # ultimi 6 mesi

# ---------------------------------------------------------------------------
# Catalogo fittizio
# ---------------------------------------------------------------------------
CATALOGO = [
    ("SKU-101", "Succhia-clitoride Aurora", "succhia-clitoride"),
    ("SKU-102", "Vibratore Rabbit Nova", "vibratori"),
    ("SKU-103", "Wand massaggiante Vela", "wand"),
    ("SKU-104", "Vibratore per coppie Duo App", "coppie-app"),
    ("SKU-105", "Anello fallico Orbit", "per-lui"),
    ("SKU-106", "Masturbatore Echo", "per-lui"),
    ("SKU-107", "Kit plug progressivo", "anale"),
    ("SKU-108", "Dildo in silicone Onda", "dildo"),
    ("SKU-109", "Palline vaginali Kegel Set", "palline"),
    ("SKU-110", "Lubrificante base acqua Fresh 100ml", "lubrificanti"),
    ("SKU-111", "Kit bondage Soft Start", "bdsm"),
    ("SKU-112", "Body in pizzo Luce", "lingerie"),
]

# Vincoli di coerenza: un lubrificante non ha la batteria, un body non ha l'app.
# Senza questi vincoli il dataset produce assurdita' che si vedono subito.
ELETTRONICI = ["SKU-101", "SKU-102", "SKU-103", "SKU-104", "SKU-105", "SKU-106"]
CON_APP = ["SKU-104", "SKU-102"]
CON_TAGLIA = ["SKU-112"]

# prodotti "critici" su cui concentriamo alcuni difetti ricorrenti,
# cosi' il clustering ha qualcosa di vero da trovare
BATTERIA_CRITICI = ["SKU-101", "SKU-103"]
APP_CRITICI = ["SKU-104"]
TAGLIA_CRITICI = ["SKU-112"]

# ---------------------------------------------------------------------------
# Template per tema. {p} = nome prodotto
# ---------------------------------------------------------------------------
TEMPLATE = {
    "logistica_consegna": [
        "Ordinato il {gg} e dopo dieci giorni il pacco non è ancora arrivato. Il tracking è fermo da una settimana.",
        "Spedizione in ritardo di oltre una settimana rispetto a quanto indicato in fase di acquisto.",
        "Il corriere ha segnato consegnato ma non ho ricevuto nulla. Ho scritto e sto ancora aspettando.",
        "Tracking mai aggiornato, non ho idea di dove sia il mio ordine. Un po' di trasparenza in più non guasterebbe.",
        "Pacco bloccato in transito da sei giorni, nessuno mi sa dire nulla.",
        "Consegna prevista in 48 ore, ne sono passate 96. Il prodotto mi serviva per un regalo.",
        "Ho pagato la spedizione express e è arrivato con gli stessi tempi di quella standard.",
        "Ordine diviso in due spedizioni senza avvisarmi, ho ricevuto solo metà di quello che ho pagato.",
        "Il {p} risulta consegnato al vicino ma nessuno mi ha avvisato, l'ho scoperto per caso.",
        "Seconda volta che un vostro ordine arriva in ritardo. La prima l'ho lasciata correre.",
    ],
    "imballo_privacy": [
        "Imballaggio davvero discreto, sulla scatola non c'è scritto niente. Su questo siete impeccabili.",
        "Doppia scatola e nessun riferimento al contenuto, per me è fondamentale visto che vivo con i miei.",
        "Però sull'estratto conto compare un nome che si capisce benissimo cosa sia. Sul sito dite il contrario.",
        "Il pacco era anonimo ma la bolla del corriere riportava la descrizione della merce.",
        "Vivo in condivisione e la discrezione era il motivo per cui ho scelto voi: promessa mantenuta.",
        "Ho chiesto se il nome che compare sulla carta è riconoscibile e non ho avuto risposta chiara.",
        "Confezione curata, si capisce che ci avete pensato. Il packaging è anche bello da vedere.",
    ],
    "reso_rimborso": [
        "Ho chiesto il reso del {p} venti giorni fa, il rimborso non è ancora arrivato.",
        "Procedura di reso poco chiara, sul sito non si capisce cosa si può restituire e cosa no.",
        "Mi avevano promesso il rimborso via mail, da allora silenzio totale.",
        "Ho ricevuto un prodotto diverso da quello ordinato e mi tocca pagare io la spedizione del reso.",
        "Reso rifiutato perché la confezione era aperta, ma come faccio a capire se il prodotto va bene senza aprirlo?",
        "Chiedo da tre settimane lo stato del rimborso, ricevo solo risposte generiche.",
        "Rimborso arrivato parziale senza nessuna spiegazione sulla differenza.",
        "Vorrei sostituire il {p} con un modello diverso, ma non trovo da nessuna parte come si fa il cambio.",
    ],
    "difetto_batteria": [
        "Il {p} non tiene la carica: dopo venti minuti è già scarico.",
        "Dopo tre settimane il {p} non si accende più, la base magnetica non aggancia bene.",
        "Il cavo di ricarica in dotazione ha smesso di funzionare quasi subito.",
        "Si scarica molto più in fretta di quanto dichiarato nella scheda. Parlavate di due ore.",
        "Il {p} si spegne da solo a metà utilizzo anche se la spia dice che è carico.",
        "La ricarica magnetica si stacca al minimo movimento, è impossibile lasciarlo in carica.",
        "Ho dovuto sostituirlo perché la batteria era gonfia. Assistenza gentile ma lentissima.",
    ],
    "difetto_prodotto": [
        "Il {p} ha smesso di vibrare dopo pochi utilizzi.",
        "Il {p} è arrivato con un difetto sul silicone, c'era proprio un taglio.",
        "Molto più rumoroso di quanto immaginassi, si sente da fuori dalla stanza.",
        "Il {p} ha un odore di plastica molto forte appena aperto, non me lo aspettavo da un silicone medicale.",
        "Uno dei pulsanti del {p} non risponde, devo premerlo cinque volte.",
        "Il {p} doveva essere impermeabile e invece dopo averlo sciacquato ha iniziato a fare le bizze.",
        "Mi è arrivato già usato, la confezione era chiaramente stata aperta.",
        "La cinghia del {p} si è strappata la seconda volta che l'ho usato.",
    ],
    "uso_prodotto": [
        "Come si carica il {p}? Nella confezione non c'era nessun foglietto di istruzioni.",
        "Le istruzioni del {p} sono solo in inglese e in cinese, in italiano niente.",
        "Non riesco a collegare il {p} all'app, il bluetooth non lo trova.",
        "L'app si disconnette in continuazione quando lo controllo a distanza. È normale?",
        "Qual è il modo corretto di pulire il {p}? Posso usare acqua e sapone o serve un detergente apposta?",
        "Il {p} si può usare con un lubrificante al silicone o rovina il materiale?",
        "Non capisco a cosa servano le diverse modalità del {p}, sul libretto non è spiegato.",
        "Come faccio a capire se il {p} è completamente carico? La spia lampeggia e basta.",
        "Si può usare il {p} sotto la doccia oppure no?",
        "Dove si conserva il {p}? Va bene lasciarlo nella custodia insieme ad altri?",
        "Il {p} ha un blocco tasti? Mi si accende da solo in borsa.",
    ],
    "scelta_preacquisto": [
        "È il mio primo acquisto di questo tipo e non so proprio da dove iniziare. Potete consigliarmi?",
        "Sto scegliendo tra il {p} e un altro modello, non capisco la differenza tra i due.",
        "Vorrei fare un regalo alla mia partner ma ho paura di sbagliare, come mi oriento?",
        "Il {p} è adatto a chi è alle prime armi o è troppo intenso?",
        "Che taglia prendo del {p}? Le misure sulla scheda non corrispondono a quelle italiane.",
        "Il {p} è silenzioso? Vivo con dei coinquilini e per me è la cosa più importante.",
        "Il materiale del {p} è compatibile con chi ha la pelle sensibile?",
        "Cerco qualcosa da usare in coppia a distanza, quale mi consigliate?",
        "Il {p} si può usare anche in gravidanza?",
        "Volevo capire se il {p} è utile per la riabilitazione del pavimento pelvico o è solo per il piacere.",
        "Sono indecisa tra due prodotti della stessa categoria, esiste una guida che li confronti?",
        "Il {p} funziona anche senza app o serve per forza lo smartphone?",
    ],
    "informazione_scheda": [
        "Nella scheda del {p} non sono indicate le dimensioni reali, dal vivo è molto più grande.",
        "Sarebbe utile indicare quanti decibel fa il {p}, la rumorosità non è scritta da nessuna parte.",
        "Le foto del {p} non rendono l'idea del colore vero.",
        "Non si capisce dalla descrizione se il {p} è ricaricabile o va a pile.",
        "Manca l'indicazione del materiale nella scheda del {p}, ho dovuto scrivervi per saperlo.",
        "Le informazioni sul prodotto sono insufficienti, ho comprato praticamente alla cieca.",
        "Sulla scheda del {p} non è chiaro cosa sia incluso nella confezione.",
        "Descrizione molto vaga, le recensioni degli altri clienti mi hanno aiutato più della scheda.",
    ],
    "assistenza": [
        "Ho scritto due volte all'assistenza e non ho ricevuto nessuna risposta.",
        "Risposte vaghe, copiate e incollate, che non risolvono il problema.",
        "Assistenza gentile ma non risolutiva, mi hanno rimandato a una pagina che non c'entrava nulla.",
        "Ho dovuto raccontare tutto da capo a tre persone diverse.",
        "Ho scritto su Instagram e mi hanno detto di mandare una mail, ho mandato la mail e nessuno risponde.",
        "Assistenza clienti davvero impeccabile, mi hanno risposto in mezza giornata e risolto tutto.",
        "Ho ricevuto una risposta chiara e senza giudizio, e per un acquisto del genere conta parecchio.",
        "Mi hanno risposto dopo cinque giorni quando sul sito dichiarate 48 ore.",
    ],
    "promo_ordini": [
        "Il codice sconto non si applica al carrello, ho provato con tre browser diversi.",
        "Ho ordinato durante la promozione e mi è stato addebitato il prezzo pieno.",
        "Non riesco a scaricare la fattura dall'area personale.",
        "Avevo diritto al Bonus Goduria ma non capisco come si richiede.",
        "Il prezzo in newsletter era diverso da quello sul sito al momento dell'acquisto.",
        "Ho pagato la spedizione anche se avevo superato la soglia della spedizione gratuita.",
        "Non mi arrivano più le mail con gli ordini, controllato anche lo spam.",
    ],
    "corsi_contenuti": [
        "Ho seguito uno dei vostri corsi e l'ho trovato fatto molto bene, chiaro e senza imbarazzo.",
        "I vostri articoli sono l'unica cosa in italiano che spieghi certe cose senza fare i moralisti.",
        "Mi piacerebbe un contenuto che spieghi come scegliere il primo prodotto, non ho trovato nulla del genere.",
        "Il video corso è ottimo ma la piattaforma è scomoda da usare da telefono.",
        "Sarebbe bello avere una guida ai materiali, ogni volta mi perdo tra silicone, ABS e TPE.",
        "Complimenti per il tono che usate, mi ha fatto sentire a mio agio a fare domande.",
    ],
    "feedback_positivo": [
        "Il {p} è esattamente come lo descrivete, qualità ottima e spedizione velocissima.",
        "Prodotto arrivato in due giorni, imballo perfetto e anonimo. Sono molto soddisfatta.",
        "Il {p} vale ogni euro speso. Comprerò ancora da voi.",
        "Ottimo acquisto, il {p} è silenzioso e ben rifinito. Consigliatissimo.",
        "Servizio impeccabile dall'ordine alla consegna, e il {p} ha superato le aspettative.",
        "Comprato per curiosità, tornata per la seconda volta. Il {p} è ottimo.",
        "Finalmente un negozio italiano serio su questo tema. Il {p} è di qualità alta.",
        "Materiale piacevole, ricarica veloce, tutto come promesso.",
    ],
}

# rating tipico per tema (min, max) sulle recensioni
RATING = {
    "logistica_consegna": (1, 2),
    "imballo_privacy": (3, 5),
    "reso_rimborso": (1, 2),
    "difetto_batteria": (1, 3),
    "difetto_prodotto": (1, 3),
    "uso_prodotto": (2, 4),
    "scelta_preacquisto": (3, 4),
    "informazione_scheda": (2, 4),
    "assistenza": (1, 4),
    "promo_ordini": (2, 3),
    "corsi_contenuti": (4, 5),
    "feedback_positivo": (5, 5),
}

# quanto pesa ogni tema nelle recensioni e nei ticket
PESI_RECENSIONI = {
    "feedback_positivo": 30,
    "logistica_consegna": 16,
    "difetto_batteria": 8,
    "difetto_prodotto": 8,
    "reso_rimborso": 9,
    "imballo_privacy": 8,
    "informazione_scheda": 7,
    "assistenza": 6,
    "uso_prodotto": 4,
    "promo_ordini": 2,
    "corsi_contenuti": 2,
}

PESI_TICKET = {
    "uso_prodotto": 22,
    "scelta_preacquisto": 20,
    "logistica_consegna": 16,
    "reso_rimborso": 13,
    "difetto_batteria": 7,
    "difetto_prodotto": 6,
    "promo_ordini": 6,
    "informazione_scheda": 4,
    "imballo_privacy": 4,
    "assistenza": 2,
}

CODE = [
    "Buongiorno, ", "Salve, ", "Ciao, ", "", "", "Buonasera, ",
]
CHIUSE = [
    " Grazie in anticipo.", " Grazie mille.", " Resto in attesa.", "", "", "",
    " Potete aiutarmi?", " Fatemi sapere per favore.",
]


def scegli_prodotto(tema, template=""):
    """Sceglie un prodotto COERENTE con quello che dice il testo.

    Se il template parla di app, il prodotto deve avere l'app; se parla di
    ricarica, deve essere elettronico; se parla di taglia, deve essere un capo.
    In piu' concentra alcuni difetti su SKU specifici, come accade nella
    realta', cosi' il clustering ha un segnale vero da trovare.
    """
    t = template.lower()
    parla_di_app = any(w in t for w in ("app", "bluetooth", "smartphone"))
    parla_di_carica = any(w in t for w in ("caric", "spia", "batteri", "cavo", "si accende", "si spegne"))
    parla_di_taglia = any(w in t for w in ("taglia", "misure sulla scheda"))

    if tema == "difetto_batteria":
        pool = BATTERIA_CRITICI if random.random() < 0.55 else ELETTRONICI
    elif parla_di_app:
        pool = APP_CRITICI if random.random() < 0.7 else CON_APP
    elif parla_di_carica:
        pool = ELETTRONICI
    elif parla_di_taglia:
        pool = CON_TAGLIA
    elif tema == "scelta_preacquisto" and random.random() < 0.2:
        pool = CON_TAGLIA
    else:
        pool = [c[0] for c in CATALOGO]

    sku = random.choice(pool)
    return next(p for p in CATALOGO if p[0] == sku)


def estrai_tema(pesi):
    temi = list(pesi.keys())
    return random.choices(temi, weights=[pesi[t] for t in temi], k=1)[0]


def componi(template, nome_prodotto, giorno):
    return template.replace("{p}", nome_prodotto).replace("{gg}", giorno.strftime("%d/%m"))


def data_casuale():
    return OGGI - timedelta(days=random.randint(0, GIORNI))


def genera_recensioni(n=460):
    righe = []
    for i in range(n):
        tema = estrai_tema(PESI_RECENSIONI)
        template = random.choice(TEMPLATE[tema])
        sku, nome, categoria = scegli_prodotto(tema, template)
        giorno = data_casuale()
        testo = componi(template, nome, giorno)
        if random.random() < 0.35:
            testo = random.choice(CODE) + testo[0].lower() + testo[1:]
        lo, hi = RATING[tema]
        righe.append({
            "id": f"REC-{1000 + i}",
            "data": giorno.isoformat(),
            "fonte": random.choices(["sito", "trustpilot"], weights=[70, 30])[0],
            "sku": sku,
            "prodotto": nome,
            "categoria": categoria,
            "rating": random.randint(lo, hi),
            "testo": testo,
            "tema_reale": tema,  # solo per validare il modello, non usato in input
        })
    return righe


def genera_ticket(n=380):
    righe = []
    canali = ["email", "chat", "instagram", "whatsapp"]
    pesi_canali = [45, 25, 20, 10]
    for i in range(n):
        tema = estrai_tema(PESI_TICKET)
        template = random.choice(TEMPLATE[tema])
        sku, nome, categoria = scegli_prodotto(tema, template)
        giorno = data_casuale()
        testo = componi(template, nome, giorno)
        testo = random.choice(CODE) + testo[0].lower() + testo[1:] + random.choice(CHIUSE)
        righe.append({
            "id": f"TCK-{2000 + i}",
            "data": giorno.isoformat(),
            "canale": random.choices(canali, weights=pesi_canali)[0],
            "sku": sku,
            "prodotto": nome,
            "categoria": categoria,
            "testo": testo.strip(),
            "tema_reale": tema,
        })
    return righe


def salva(righe, percorso):
    with open(percorso, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(righe[0].keys()))
        w.writeheader()
        w.writerows(righe)
    print(f"scritto {percorso} ({len(righe)} righe)")


if __name__ == "__main__":
    salva(genera_recensioni(), DATA_DIR / "recensioni.csv")
    salva(genera_ticket(), DATA_DIR / "ticket.csv")

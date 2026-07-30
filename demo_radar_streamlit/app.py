from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


st.set_page_config(page_title="Radar cyber France", page_icon="◉", layout="wide")

DATA_DIR = Path(__file__).parent / "data"


@st.cache_data
def load_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / filename, dtype=str, keep_default_na=False)


def clean_text(value: object) -> str:
    return str(value).replace("Ã©", "é").replace("Ã¨", "è").replace("Ãª", "ê").replace("Ã ", "à").replace("â€™", "’").replace("â€œ", "“").replace("â€", "”")


@st.cache_data
def find_relevant_signals(question: str, corpus: pd.DataFrame) -> pd.DataFrame:
    """Recherche des extraits du corpus, sans appel Ã  un service externe."""
    columns = ["organisation_concernee", "categorie", "resume_neutre", "theme", "sous_theme", "passage_justificatif"]
    documents = corpus[columns].fillna("").agg(" ".join, axis=1)
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(documents)
    scores = linear_kernel(vectorizer.transform([question]), matrix).flatten()
    results = corpus.copy()
    results["score_pertinence"] = scores
    return results[results["score_pertinence"] > 0].sort_values("score_pertinence", ascending=False).head(5)


entities = load_csv("entites_publiques.csv")
signals = load_csv("signaux_publics.csv")
services = load_csv("services_publics.csv")
relations = load_csv("relations_publiques.csv")

for frame in (entities, signals, services, relations):
    for column in frame.columns:
        frame[column] = frame[column].map(clean_text)

st.title("Radar stratégique cyber France")
st.caption("Prototype de démonstration — corpus public, à valider")

with st.expander("Poser une question au corpus", expanded=True):
    st.write("Formulez votre question naturellement. L'application affiche seulement les extraits pertinents du corpus : aucune information n'est inventÃ©e et aucune donnÃ©e n'est envoyÃ©e Ã  un service externe.")
    question = st.text_input("Exemple : quels acteurs proposent des services SOC ou MDR pour les MSP ?", key="question")
    if question:
        matches = find_relevant_signals(question, signals)
        if matches.empty:
            st.info("Aucun passage suffisamment proche n'a Ã©tÃ© trouvÃ©. Essayez avec le nom d'un acteur ou des mots plus simples.")
        else:
            st.success(f"{len(matches)} passages pertinents trouvÃ©s dans le corpus")
            for _, item in matches.iterrows():
                st.markdown(f"**{item['organisation_concernee']} â€” {item['categorie']}**")
                st.write(item["resume_neutre"])
                st.caption(f"RÃ©fÃ©rence : {item['id_signal']} Â· {item['id_source_exacte']} Â· confiance : {item['niveau_confiance']}")


with st.sidebar:
    st.header("Explorer le corpus")
    query = st.text_input("Rechercher un acteur, un service ou un sujet")
    organisations = sorted(x for x in signals["organisation_concernee"].unique() if x)
    selected_orgs = st.multiselect("Acteurs", organisations)
    categories = sorted(x for x in signals["categorie"].unique() if x)
    selected_categories = st.multiselect("Types de signal", categories)

filtered = signals.copy()
if selected_orgs:
    filtered = filtered[filtered["organisation_concernee"].isin(selected_orgs)]
if selected_categories:
    filtered = filtered[filtered["categorie"].isin(selected_categories)]
if query:
    searchable = filtered[["organisation_concernee", "categorie", "resume_neutre", "theme", "sous_theme"]].fillna("").agg(" ".join, axis=1)
    filtered = filtered[searchable.str.contains(query, case=False, na=False)]

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Signaux", len(filtered))
metric_2.metric("Acteurs observés", filtered["organisation_concernee"].nunique())
metric_3.metric("Services documentés", len(services))
metric_4.metric("Relations documentées", len(relations))

tab_signals, tab_actors, tab_services, tab_relations = st.tabs(
    ["Signaux", "Acteurs", "Services", "Relations"]
)

with tab_signals:
    st.subheader("Signaux de marché")
    display = filtered[["organisation_concernee", "categorie", "resume_neutre", "nature_information", "niveau_confiance", "statut_validation"]].rename(
        columns={
            "organisation_concernee": "Acteur",
            "categorie": "Catégorie",
            "resume_neutre": "Résumé",
            "nature_information": "Nature",
            "niveau_confiance": "Confiance",
            "statut_validation": "Validation",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True, height=520)
    st.download_button("Télécharger les signaux filtrés (CSV)", display.to_csv(index=False).encode("utf-8-sig"), "signaux_filtres.csv", "text/csv")

with tab_actors:
    st.subheader("Acteurs du corpus")
    actor_view = entities[["nom_canonique", "type_organisation", "pays", "region_france", "role_chaine_valeur", "statut_qualification_msp"]].rename(
        columns={"nom_canonique": "Acteur", "type_organisation": "Type", "pays": "Pays", "region_france": "Région", "role_chaine_valeur": "Rôle", "statut_qualification_msp": "Qualification MSP"}
    )
    st.dataframe(actor_view, use_container_width=True, hide_index=True, height=520)

with tab_services:
    st.subheader("Services et offres")
    service_view = services[["nom_service", "domaine", "mode_livraison", "couverture_horaire", "segments_cibles", "geographie", "statut_validation"]].rename(
        columns={"nom_service": "Service", "domaine": "Domaine", "mode_livraison": "Livraison", "couverture_horaire": "Couverture", "segments_cibles": "Segments", "geographie": "Géographie", "statut_validation": "Validation"}
    )
    st.dataframe(service_view, use_container_width=True, hide_index=True, height=520)

with tab_relations:
    st.subheader("Relations entre acteurs")
    relation_view = relations[["entite_source", "type_relation", "entite_cible", "organisation_concernee", "nature_information", "statut_validation"]].rename(
        columns={"entite_source": "Source", "type_relation": "Relation", "entite_cible": "Cible", "organisation_concernee": "Organisation", "nature_information": "Nature", "statut_validation": "Validation"}
    )
    st.dataframe(relation_view, use_container_width=True, hide_index=True, height=520)

st.divider()
st.caption("Les contenus sont issus du corpus public existant. Ils servent à explorer des pistes de veille et ne constituent pas des conclusions définitives.")

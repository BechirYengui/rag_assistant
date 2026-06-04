# RAG Assistant

Assistant de recherche augmentée (RAG) qui répond en langage naturel à des
questions posées sur une base documentaire. Le projet couvre toute la chaîne :
ingestion et découpage des documents, génération d'embeddings, recherche
vectorielle, puis génération d'une réponse sourcée par un LLM, le tout exposé
via une API REST.

Stack technique : Python, FastAPI (asynchrone), embeddings (fastembed en local
ou Voyage AI), base vectorielle pgvector (PostgreSQL), et l'API Claude pour la
génération des réponses.


## Sommaire

1. Fonctionnalités
2. Architecture
3. Structure du projet
4. Prérequis
5. Installation
6. Configuration
7. Lancement
8. Utilisation de l'API
9. Fonctionnement interne
10. Tests
11. Dépannage
12. Pistes d'amélioration


## 1. Fonctionnalités

- Ingestion de documents au format texte, Markdown ou PDF (envoi direct ou
  upload de fichier).
- Découpage automatique des documents en segments (chunks) avec recouvrement.
- Génération d'embeddings via un modèle local (aucune clé requise) ou Voyage AI.
- Stockage et recherche des vecteurs dans PostgreSQL grâce à l'extension
  pgvector, avec un index ANN de type HNSW.
- Réponses générées par Claude, accompagnées des sources utilisées et de leur
  score de similarité, pour une traçabilité complète.
- Réponse classique en JSON ou réponse en streaming (Server-Sent Events).
- Configuration entièrement pilotée par variables d'environnement.


## 2. Architecture

Le flux se déroule en deux phases.

Phase d'indexation (hors ligne) :

```
  Documents          Decoupage          Embeddings         pgvector
 (txt/md/pdf)  ->    en chunks    ->    (vecteurs)   ->   (PostgreSQL)
```

Phase de question/reponse (en ligne) :

```
  Question  ->  Embedding  ->  Recherche k plus proches voisins (cosinus)
                                           |
                                           v
                                 Contexte (chunks pertinents)
                                           |
                                           v
                              Claude  ->  Reponse + citations
```

Chaque réponse renvoie le texte généré ainsi que la liste des chunks
récupérés, ce qui permet de vérifier d'où vient chaque information.


## 3. Structure du projet

```
rag-assistant/
  app/
    config.py            Configuration (variables d'environnement)
    db.py                Moteur SQLAlchemy async, bootstrap pgvector et index HNSW
    models.py            Modeles ORM Document et Chunk
    schemas.py           Schemas Pydantic (contrat de l'API)
    chunking.py          Decoupage recursif avec recouvrement
    embeddings.py        Fournisseurs d'embeddings (fastembed ou Voyage)
    retrieval.py         Recherche vectorielle k-NN via pgvector
    llm.py               Generation de la reponse citee avec Claude
    ingestion.py         Pipeline d'ingestion complet
    rag.py               Orchestration recherche + generation
    main.py              Point d'entree FastAPI
    routers/
      documents.py       Endpoints de gestion des documents
      query.py           Endpoints de question/reponse
  scripts/
    ingest_cli.py        Script d'ingestion de fichiers en lot
  tests/
    test_chunking.py     Tests du decoupage
    test_api.py          Tests du pipeline et des routes
  sample_docs/           Documents d'exemple
  docker-compose.yml     Base pgvector + API
  Dockerfile             Image de l'API
  Makefile               Raccourcis de commandes
  requirements.txt       Dependances d'execution
  requirements-dev.txt   Dependances de developpement
  .env.example           Modele de configuration
```


## 4. Prérequis

- Python 3.10 ou supérieur.
- Docker et Docker Compose (pour la base pgvector).
- Une clé API Anthropic (variable ANTHROPIC_API_KEY) pour générer les réponses.
  Les embeddings par défaut tournent en local et ne nécessitent aucune clé.


## 5. Installation

Cloner le dépôt puis installer les dépendances :

```
cd rag-assistant
make dev
```

`make dev` installe les dépendances d'exécution et de développement. Pour ne
poser que les dépendances d'exécution : `make install`.

Copier ensuite le modèle de configuration et renseigner la clé Claude :

```
cp .env.example .env
```

Éditer `.env` et remplir au minimum `ANTHROPIC_API_KEY`.


## 6. Configuration

Toutes les options se définissent par variables d'environnement (ou dans le
fichier `.env`). Les principales :

| Variable | Défaut | Description |
|---|---|---|
| ANTHROPIC_API_KEY | (vide) | Clé API Claude, requise pour /query |
| LLM_MODEL | claude-opus-4-8 | Modèle utilisé pour la génération |
| LLM_EFFORT | high | Profondeur de raisonnement (low, medium, high, max) |
| DATABASE_URL | postgresql+asyncpg://rag:rag@localhost:5432/rag | URL de la base |
| EMBEDDING_PROVIDER | fastembed | fastembed (local) ou voyage (hébergé) |
| EMBEDDING_MODEL | BAAI/bge-small-en-v1.5 | Modèle d'embeddings |
| EMBEDDING_DIM | 384 | Dimension des vecteurs |
| CHUNK_SIZE | 1000 | Taille cible d'un chunk (caractères) |
| CHUNK_OVERLAP | 150 | Recouvrement entre chunks (caractères) |
| RETRIEVAL_TOP_K | 5 | Nombre de chunks récupérés par question |
| RETRIEVAL_MIN_SIMILARITY | 0.2 | Seuil de similarité minimal |

Attention : `EMBEDDING_DIM` doit correspondre à la dimension réelle du modèle
d'embeddings, car c'est la largeur de la colonne vectorielle en base. Le modèle
bge-small produit des vecteurs de dimension 384, voyage-3 de dimension 1024.
Changer de modèle implique de recréer la base.


## 7. Lancement

Démarrer la base pgvector :

```
make db
```

Lancer l'API en local (rechargement automatique) :

```
make run
```

L'API est alors disponible sur http://localhost:8000 et la documentation
interactive sur http://localhost:8000/docs.

Indexer les documents d'exemple :

```
make ingest
```

Tout lancer en une seule commande avec Docker (base et API) :

```
ANTHROPIC_API_KEY=sk-ant-xxx docker compose up --build
```


## 8. Utilisation de l'API

Liste des endpoints :

| Méthode | Endpoint | Description |
|---|---|---|
| POST | /documents | Ingestion d'un document texte (JSON) |
| POST | /documents/upload | Ingestion d'un fichier txt, md ou pdf |
| GET | /documents | Liste des documents indexés |
| DELETE | /documents/{id} | Suppression d'un document et de ses chunks |
| POST | /query | Question, réponse sourcée en JSON |
| POST | /query/stream | Question, réponse en streaming (SSE) |
| GET | /health | Statut et configuration active |

Exemple d'ingestion d'un texte :

```
curl -X POST localhost:8000/documents \
  -H 'content-type: application/json' \
  -d '{
    "content": "Le texte du document a indexer...",
    "source": "note-interne.txt",
    "title": "Note interne"
  }'
```

Exemple de question :

```
curl -X POST localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{
    "question": "Quelle difference entre les index HNSW et IVFFlat ?"
  }'
```

Exemple de réponse :

```json
{
  "question": "Quel operateur pgvector utiliser pour la similarite cosinus ?",
  "answer": "Pour la similarite cosinus, utilisez l'operateur <=> [1]. La similarite cosinus vaut un moins la distance cosinus [1].",
  "sources": [
    {
      "citation": 1,
      "source": "pgvector.md",
      "title": "pgvector",
      "chunk_index": 0,
      "similarity": 0.83,
      "content": "..."
    }
  ],
  "model": "claude-opus-4-8"
}
```

Le marqueur [1] dans la réponse renvoie directement à la source numéro 1 de la
liste, ce qui rend chaque affirmation vérifiable.


## 9. Fonctionnement interne

Découpage (chunking). Le texte est d'abord normalisé puis découpé de manière
récursive en essayant de couper sur la frontière naturelle la plus large qui
tient dans la fenêtre cible : paragraphe, puis ligne, puis phrase, puis mot.
Les fragments sont ensuite regroupés en fenêtres d'environ CHUNK_SIZE
caractères, chaque fenêtre partageant CHUNK_OVERLAP caractères avec la
précédente pour ne pas perdre le contexte aux jonctions.

Embeddings. Une abstraction commune (EmbeddingProvider) permet de basculer
entre un modèle local (fastembed, basé sur ONNX, sans clé) et Voyage AI
(hébergé). Les imports lourds sont différés au moment de la construction, ce
qui permet de charger le reste de l'application et de lancer les tests sans
installer ces dépendances.

Recherche. La question est transformée en vecteur, puis pgvector récupère les
chunks les plus proches au sens de la distance cosinus, via l'index HNSW. La
similarité est calculée comme un moins la distance, et les résultats sous le
seuil minimal sont écartés.

Génération. Les chunks retenus sont assemblés en un bloc de contexte numéroté.
Le prompt système impose à Claude de ne répondre qu'à partir de ce contexte, de
citer chaque affirmation avec un marqueur entre crochets, et d'indiquer
clairement lorsqu'il n'a pas l'information plutôt que d'inventer. La génération
utilise le mode adaptive thinking, le paramètre effort et le streaming.

Robustesse. L'ingestion d'un document et de ses chunks se fait dans une seule
transaction : en cas d'erreur, rien n'est indexé partiellement. Le schéma de la
base (extension, tables, index) est créé automatiquement au démarrage.


## 10. Tests

```
make test
```

Les tests du découpage et du pipeline (formatage du contexte, attribution des
citations, validation des requêtes, endpoint /health) s'exécutent sans base
PostgreSQL ni clé Claude, grâce à des doublures et au transport ASGI en
mémoire.


## 11. Dépannage

- Erreur de dimension à l'insertion : la valeur EMBEDDING_DIM ne correspond pas
  au modèle d'embeddings. Aligner les deux et recréer la base.
- 500 sur /query avec un message sur ANTHROPIC_API_KEY : la clé Claude n'est pas
  renseignée dans l'environnement.
- La base ne démarre pas : vérifier que le port 5432 est libre et que Docker
  est lancé.
- Le premier appel est lent : au premier usage, le modèle d'embeddings local est
  téléchargé puis mis en cache.


## 12. Pistes d'amélioration

- Reranking des chunks récupérés avant génération.
- Recherche hybride (lexicale et vectorielle).
- Authentification et quotas par utilisateur.
- Cache des réponses fréquentes.
- Jeu d'évaluation pour mesurer la qualité des réponses.

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
10. Optimisation des tokens Claude
11. Tests
12. Dépannage
13. Pistes d'amélioration


## 1. Fonctionnalités

- Ingestion de documents au format texte, Markdown ou PDF (envoi direct ou
  upload de fichier).
- Découpage automatique des documents en segments (chunks) avec recouvrement.
- Génération d'embeddings via un modèle local (aucune clé requise) ou Voyage AI.
- Stockage et recherche des vecteurs dans PostgreSQL grâce à l'extension
  pgvector, avec un index ANN de type HNSW.
- Reranking MMR des résultats pour diversifier le contexte et écarter les
  passages quasi identiques.
- Ingestion idempotente : un document au contenu identique n'est pas réindexé.
- Réponses générées par Claude, accompagnées des sources utilisées et de leur
  score de similarité, pour une traçabilité complète.
- Réponse classique en JSON ou réponse en streaming (Server-Sent Events).
- Maîtrise du coût en tokens (budget de contexte, cache de réponses, suivi de
  l'usage) et configuration entièrement pilotée par variables d'environnement.


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
| LLM_EFFORT | low | Profondeur de raisonnement (low, medium, high, max) |
| LLM_THINKING | disabled | disabled (le moins cher) ou adaptive |
| LLM_MAX_TOKENS | 1024 | Longueur maximale de la réponse |
| MAX_CONTEXT_TOKENS | 1200 | Plafond du contexte envoyé à Claude (tokens estimés) |
| ANSWER_CACHE_SIZE | 256 | Taille du cache de réponses (0 pour désactiver) |
| DATABASE_URL | postgresql+asyncpg://rag:rag@localhost:5432/rag | URL de la base |
| EMBEDDING_PROVIDER | fastembed | fastembed (local) ou voyage (hébergé) |
| EMBEDDING_MODEL | BAAI/bge-small-en-v1.5 | Modèle d'embeddings |
| EMBEDDING_DIM | 384 | Dimension des vecteurs |
| CHUNK_SIZE | 1000 | Taille cible d'un chunk (caractères) |
| CHUNK_OVERLAP | 150 | Recouvrement entre chunks (caractères) |
| RETRIEVAL_TOP_K | 5 | Nombre de chunks récupérés par question |
| RETRIEVAL_MIN_SIMILARITY | 0.2 | Seuil de similarité minimal |
| RERANK | mmr | Reranking des résultats (mmr ou none) |
| MMR_LAMBDA | 0.5 | Arbitrage MMR pertinence/diversité (1 = pertinence pure) |
| RERANK_FETCH_MULTIPLIER | 4 | Taille du vivier de candidats avant reranking (x top_k) |

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
| GET | /health | Liveness et configuration active (toujours 200) |
| GET | /health/ready | Readiness : 200 si la base est joignable, sinon 503 |

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
  "model": "claude-opus-4-8",
  "usage": { "input_tokens": 312, "output_tokens": 28, "cache_read_input_tokens": 0 },
  "cached": false
}
```

Le marqueur [1] dans la réponse renvoie directement à la source numéro 1 de la
liste, ce qui rend chaque affirmation vérifiable. Le bloc usage indique le coût
en tokens de l'appel (voir section 10), et cached vaut true si la réponse vient
du cache (zéro token).


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

Ingestion idempotente. Avant tout calcul, on prend l'empreinte SHA-256 du
contenu normalisé. Si un document identique a déjà été indexé, on renvoie
l'existant sans recalculer les embeddings ni créer de doublon. Réimporter le
même fichier ne coûte donc rien et ne pollue pas la base.

Recherche. La question est transformée en vecteur, puis pgvector récupère les
chunks les plus proches au sens de la distance cosinus, via l'index HNSW. La
similarité est calculée comme un moins la distance, et les résultats sous le
seuil minimal sont écartés.

Reranking MMR. La recherche par plus proches voisins tend à remonter des
passages qui se ressemblent tous (la même information formulée plusieurs fois),
ce qui gaspille des tokens et réduit la couverture. On récupère donc un vivier
plus large (top_k multiplié par RERANK_FETCH_MULTIPLIER), puis l'algorithme
Maximal Marginal Relevance sélectionne un sous-ensemble à la fois pertinent
pour la question et diversifié. Le score d'un candidat est
MMR_LAMBDA fois sa pertinence moins (1 moins MMR_LAMBDA) fois sa ressemblance
maximale avec les passages déjà retenus. Le calcul est en Python pur, sans
dépendance, et le vivier est borné par RERANK_MAX_POOL.

Génération. Les chunks retenus sont assemblés en un bloc de contexte numéroté.
Le prompt système impose à Claude de ne répondre qu'à partir de ce contexte, de
citer chaque affirmation avec un marqueur entre crochets, et d'indiquer
clairement lorsqu'il n'a pas l'information plutôt que d'inventer. Le mode de
raisonnement (thinking), le paramètre effort et la taille de réponse sont
réglables pour arbitrer entre qualité et coût en tokens (voir section 10).

Robustesse. L'ingestion d'un document et de ses chunks se fait dans une seule
transaction : en cas d'erreur, rien n'est indexé partiellement. Le schéma de la
base (extension, tables, index) est créé automatiquement au démarrage.


## 10. Optimisation des tokens Claude

Sur un système RAG, le coût Claude vient de trois postes : le prompt système,
le contexte récupéré (les chunks, c'est le plus gros et le plus variable), et la
réponse générée (texte plus éventuels tokens de raisonnement). Plusieurs leviers
sont intégrés pour réduire ce coût, et tous sont réglables par variables
d'environnement.

1. Raisonnement désactivé par défaut (LLM_THINKING=disabled). Pour une réponse
   extractive à partir d'un contexte fourni, le raisonnement étendu est souvent
   superflu. Le désactiver supprime entièrement les tokens de thinking. Une
   consigne « réponds uniquement par la réponse finale » est ajoutée au prompt
   système dans ce mode, pour éviter que le modèle ne déverse son raisonnement
   dans la réponse visible. Pour un corpus plus difficile, passer à
   LLM_THINKING=adaptive (meilleure qualité, plus de tokens).

2. Effort bas par défaut (LLM_EFFORT=low). Le paramètre effort règle la
   profondeur de raisonnement et la verbosité. « low » convient au Q&A
   extractif ; « high » ou « max » se réservent aux questions complexes.

3. Budget de contexte (MAX_CONTEXT_TOKENS, défaut 1200). Avant l'appel à Claude,
   les chunks sont ajoutés par ordre de pertinence décroissante jusqu'à
   atteindre ce plafond, puis on s'arrête. Le coût en tokens d'entrée est donc
   borné quel que soit top_k. Le chunk le plus pertinent est toujours conservé,
   pour ne jamais répondre avec un contexte vide. L'estimation du nombre de
   tokens se fait par une heuristique (caractères / 4) afin de ne pas dépenser
   un appel réseau juste pour mesurer.

4. Plafond de réponse (LLM_MAX_TOKENS, défaut 1024). Borne dure sur la longueur
   de la réponse : les réponses RAG sont courtes, inutile de prévoir large.

5. Cache de réponses (ANSWER_CACHE_SIZE, défaut 256). Un cache LRU en mémoire,
   indexé par (question, chunks récupérés, configuration du modèle), renvoie la
   réponse déjà calculée sans appeler Claude. Une question identique reposée
   coûte alors zéro token. Mettre 0 pour le désactiver. En production multi
   worker, remplacer par un cache partagé (Redis) derrière la même interface.

6. Mesure du coût. Chaque réponse de l'API expose un objet usage
   (input_tokens, output_tokens, cache_read_input_tokens) et un booléen cached.
   Le coût est aussi journalisé côté serveur. On peut ainsi vérifier l'effet de
   chaque réglage plutôt que de l'estimer.

Remarque sur le prompt caching. Le cache de prompt côté API ne réécrit que les
préfixes stables et volumineux (au moins 4096 tokens sur Opus). Ici le prompt
système est court et le contexte change à chaque question : il n'y a quasiment
pas de préfixe réutilisable, donc le prompt caching apporte peu. Le cache de
réponses applicatif (point 5) est le mécanisme pertinent pour ce cas d'usage.

Exemple de réponse, avec le bloc usage :

```json
{
  "answer": "Pour la similarite cosinus, utilisez <=> [1].",
  "sources": [ ... ],
  "model": "claude-opus-4-8",
  "usage": { "input_tokens": 312, "output_tokens": 28, "cache_read_input_tokens": 0 },
  "cached": false
}
```

Réglages rapides selon le besoin :

| Objectif | Réglage |
|---|---|
| Coût minimal | LLM_THINKING=disabled, LLM_EFFORT=low, MAX_CONTEXT_TOKENS bas |
| Équilibre | LLM_EFFORT=medium, MAX_CONTEXT_TOKENS=1200 |
| Qualité maximale | LLM_THINKING=adaptive, LLM_EFFORT=high, contexte plus large |


## 11. Tests

```
make test
```

Les tests couvrent le découpage, le pipeline (formatage du contexte,
attribution des citations, validation des requêtes, endpoints /health et
/health/ready, erreur 503 quand la clé Claude manque), l'optimisation des
tokens (budget de contexte, cache de réponses) et le reranking (sélection MMR,
empreinte d'ingestion). Ils s'exécutent sans base
PostgreSQL ni clé Claude, grâce à des doublures et au transport ASGI en
mémoire.


## 12. Dépannage

- Erreur de dimension à l'insertion : la valeur EMBEDDING_DIM ne correspond pas
  au modèle d'embeddings. Aligner les deux et recréer la base.
- 503 sur /query avec un message sur ANTHROPIC_API_KEY : la clé Claude n'est pas
  renseignée dans l'environnement. L'API renvoie un code 503 explicite (et non
  une erreur 500 opaque) tant que la clé manque.
- La base ne démarre pas : vérifier que le port 5432 est libre et que Docker
  est lancé.
- Le premier appel est lent : au premier usage, le modèle d'embeddings local est
  téléchargé puis mis en cache.
- Colonne content_hash absente sur une base déjà créée avant cette version : le
  schéma est créé au démarrage mais les colonnes ajoutées ne sont pas
  rétro-appliquées. Recréer la base (docker compose down -v puis up) ou ajouter
  la colonne à la main. Pour un suivi propre des évolutions de schéma, voir les
  pistes d'amélioration.


## 13. Pistes d'amélioration

- Recherche hybride (lexicale et vectorielle) combinée au reranking MMR.
- Migrations de schéma versionnées (Alembic) au lieu de la création au démarrage.
- Cache de réponses partagé (Redis) pour un déploiement multi worker.
- Authentification et quotas par utilisateur.
- Jeu d'évaluation pour mesurer la qualité des réponses et régler MMR_LAMBDA.

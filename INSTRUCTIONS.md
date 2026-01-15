# 🚀 Instructions d'utilisation - TP RAG

## 📋 Prérequis

### Installation locale

```bash
# Cloner le dépôt
git clone https://github.com/vincentmartin/tp-rag-student-version.git
cd tp-rag-student-version

# Installer les dépendances
pip install -r requirements.txt
```

### Installation sur Google Colab

Le notebook gère automatiquement l'installation. Exécutez simplement la première cellule.

## 🔧 Configuration d'Ollama

### Sur machine locale

```bash
# Installation d'Ollama
curl https://ollama.ai/install.sh | sh

# Lancement du serveur
ollama serve

# Dans un autre terminal, télécharger le modèle
ollama pull qwen2.5:3b
```

### Sur Google Colab

Exécutez ces commandes dans le terminal Colab (via colab-xterm) :

```bash
curl https://ollama.ai/install.sh | sh
ollama serve &
ollama pull qwen2.5:3b
```

## 📊 Structure du projet

```
tp-rag-student-version/
├── TP.ipynb                    # Notebook principal avec tous les exercices
├── requirements.txt            # Dépendances Python
├── README.md                   # Consignes du TP
├── INSTRUCTIONS.md            # Ce fichier
├── data/                      # Documents à indexer
│   └── arxiv/                 # PDFs scientifiques
├── chroma_db/                 # Base vectorielle (créée automatiquement)
└── multi_agent_data/          # Ressources pour exercice 8
    ├── COURS_MULTI_AGENTS_DATA.md
    └── notebooks/
```

## 🎯 Exercices implémentés

### ✅ Exercice 1 : Indexation
- Chargement des PDFs avec PyPDFDirectoryLoader
- Découpage en chunks (1000 tokens, overlap 200)
- Embeddings avec multilingual-e5-base
- Stockage dans ChromaDB

### ✅ Exercice 2 : Interrogation
- Fonction `search_documents()` avec scores de similarité
- Affichage formaté des résultats

### ✅ Exercice 3 : Prompt Template
- Template optimisé pour RAG
- Instructions claires pour l'utilisation du contexte

### ✅ Exercice 4 : Chaîne RAG
- Intégration Ollama/Qwen
- Chaîne complète avec retriever + LLM
- Fonction `ask_question()` pour tester

### ✅ Exercice 5 : Mémoire
- Gestion de l'historique avec ChatMessageHistory
- RunnableWithMessageHistory pour le suivi
- Fonction `chat_with_memory()` pour conversations

### ✅ Exercice 6 : Résumé de documents
- Fonction `summarize_document()` avec map_reduce
- Support de documents complets

### ✅ Exercice 7 : IHM Gradio
- Interface complète avec historique
- Gestion des sessions
- Design moderne et intuitif

### 📚 Exercice 8 : Évaluation
- À réaliser dans le notebook L6
- Intégration RAG local + DuckDB
- Évaluation avec TruLens

### 🌐 Exercice 9 : Recherche Web (Optionnel)
- Alternative DuckDuckGo à Tavily

## 🎮 Utilisation

### Mode Notebook

```python
# 1. Exécuter les cellules dans l'ordre
# 2. Tester la recherche
results = search_documents("What is RAG?", k=3)

# 3. Poser une question
response = ask_question("Explain multi-agent systems")

# 4. Conversation avec mémoire
chat_with_memory("What are the challenges?", session_id="my_session")
chat_with_memory("Can you elaborate?", session_id="my_session")

# 5. Résumer un document
summarize_document("data/arxiv/Complex_QA_and_language_models_hybrid_architectures_Survey.pdf")
```

### Mode Interface Gradio

Exécutez la cellule finale pour lancer l'interface web.
Un lien sera généré (avec `share=True` sur Colab, vous aurez un lien public).

## ⚙️ Configuration avancée

### Modifier le chunk size

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # Augmenter pour plus de contexte
    chunk_overlap=300,
    length_function=len,
)
```

### Changer le nombre de documents récupérés

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}  # Plus de documents
)
```

### Utiliser un modèle différent

```python
llm = ChatOllama(
    model="llama3.1:8b",  # Autre modèle
    temperature=0.2,  # Plus de créativité
)
```

## 🐛 Résolution de problèmes

### Erreur "Ollama not found"

```bash
# Vérifier qu'Ollama est installé
which ollama

# Vérifier que le serveur tourne
curl http://localhost:11434/api/tags
```

### Erreur de mémoire (OOM)

- Utilisez un modèle plus petit : `qwen2.5:1.5b`
- Réduisez le nombre de chunks : `k=2`
- Réduisez le chunk_size : `800`

### ChromaDB déjà existant

```python
# Supprimer et recréer
import shutil
shutil.rmtree("chroma_db")
```

### GPU non détecté

```python
# Pour les embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",
    model_kwargs={'device': 'cuda'}  # Forcer CUDA
)
```

## 📊 Métriques de performance

### Temps d'indexation typique
- 10 PDFs (~100 pages) : ~2-3 minutes
- Dépend de la machine et du CPU

### Temps de réponse
- Recherche vectorielle : <1 seconde
- Génération LLM : 3-10 secondes (selon modèle)
- Total : ~5-15 secondes par question

## 🎓 Ressources supplémentaires

- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [ChromaDB Guide](https://docs.trychroma.com/getting-started)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [Gradio Guide](https://www.gradio.app/guides/quickstart)
- [TruLens RAG Evaluation](https://www.trulens.org/trulens_eval/getting_started/quickstarts/quickstart/)

## 💡 Conseils pour le rendu

1. **Code propre** : Respecter PEP 8, commentaires clairs
2. **Documentation** : Docstrings pour toutes les fonctions
3. **Tests** : Montrer que chaque exercice fonctionne
4. **Évaluation** : L'exercice 8 est important pour la note
5. **Créativité** : Ajouter des fonctionnalités bonus

## 📧 Support

Pour toute question, référez-vous au README.md principal ou contactez votre enseignant.

---

**Bon TP ! 🚀**

# Exercice 8 : Évaluation Multi-Agent Local (100% Gratuit)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_REPO/exercice_8/Exercice_8_Evaluation.ipynb)

Système multi-agent local basé sur le [cours DeepLearning.AI](https://www.deeplearning.ai/short-courses/building-and-evaluating-data-agents/) - **Groq** (gratuit) au lieu de Snowflake/OpenAI (payants).

**🌐 Compatible Google Colab** - Voir [COLAB.md](COLAB.md) pour les instructions spécifiques

## 🚀 Démarrage rapide

### Option A: Google Colab (aucune installation)

1. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_REPO/exercice_8/Exercice_8_Evaluation.ipynb)
2. Ajoutez votre `GROQ_API_KEY` dans Colab Secrets (icône 🔑)
3. Uploadez `helper_local.py` et `prompts_local.py`
4. Exécutez toutes les cellules

📖 **Guide détaillé Colab**: [COLAB.md](COLAB.md)

### Option B: Installation locale (5 minutes)

#### 1. Installation

```bash
cd M2/recherche/tp-rag-student-version/exercice_8
python -m venv venv
source venv/bin/activate  # Linux/Mac (Windows: venv\Scripts\activate)
pip install -r requirements.txt
```

#### 2. Configuration API

```bash
# Obtenir clé gratuite sur https://console.groq.com/
cp env.template .env
nano .env  # Ajouter: GROQ_API_KEY=gsk_...
```

#### 3. Lancer

```bash
jupyter notebook Exercice_8_Evaluation.ipynb
# Ou: code Exercice_8_Evaluation.ipynb (VS Code)
```

## 📖 Exemples de requêtes

```python
# RAG - Documents
"What is Retrieval Augmented Generation?"

# SQL - Données structurées  
"What are our top 3 deals by value?"

# Mixte - Visualisation
"Show me a bar chart of deal values by product line."
```

Dashboard TruLens: `http://localhost:8006` (lancé automatiquement)

## 🎯 Fonctionnalités

- **RAG hiérarchique**: Petits chunks (recherche) + grands parents (contexte)
- **7 métriques TruLens**: RAG Triad (3) + GPA (4)
- **Inline evaluations**: Feedback temps réel pour auto-correction
- **Multi-agent**: Planner → Executor → Researchers → Synthesizer

## 🔧 Structure

```
exercice_8/
├── Exercice_8_Evaluation.ipynb  # ← Lancez ce notebook
├── helper_local.py              # Agents, outils, évaluations
├── prompts_local.py             # Prompts planner/executor
├── requirements.txt             # Dépendances
└── env.template                 # Template API keys
```

## 📚 Documentation

- **[QUICK_START.md](QUICK_START.md)** - Guide rapide détaillé
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Diagrammes visuels
- **[GRATUITÉ.md](GRATUITÉ.md)** - Pourquoi OpenAI n'est pas nécessaire

## 🐛 Problèmes courants

```bash
# API Key not found
cat .env  # Vérifier GROQ_API_KEY

# ChromaDB error
rm -rf ../chroma_db  # Supprimer et relancer

# Port occupé
# Dans le notebook: port = 8007
```

## 📊 vs Cours original

| Composant | L6 Original | Exercice 8 |
|-----------|-------------|------------|
| LLM | OpenAI GPT-4 | Groq llama-3.1 |
| RAG | Snowflake Cortex | ChromaDB local |
| SQL | Snowflake | DuckDB local |
| **Coût** | **$2-5/session** | **$0** 🎉 |

---

**Cours**: [Building and Evaluating Data Agents](https://www.deeplearning.ai/short-courses/building-and-evaluating-data-agents/) (DeepLearning.AI)

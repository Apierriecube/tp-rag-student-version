#!/usr/bin/env python3
"""
Script de test rapide du système RAG.
Permet de tester le système sans lancer le notebook complet.

Usage:
    python test_rag.py "Your question here"
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


class RAGSystem:
    """Système RAG simple pour tests rapides."""
    
    def __init__(self, data_path="data/arxiv", chroma_path="chroma_db"):
        self.data_path = data_path
        self.chroma_path = chroma_path
        self.vectorstore = None
        self.rag_chain = None
        
    def setup(self):
        """Initialise le système RAG."""
        print("🚀 Initialisation du système RAG...")
        
        # Vérifier si la base existe déjà
        if Path(self.chroma_path).exists():
            print(f"✅ Base vectorielle existante trouvée dans {self.chroma_path}")
            self._load_existing_vectorstore()
        else:
            print(f"📚 Création d'une nouvelle base vectorielle...")
            self._create_vectorstore()
        
        # Créer la chaîne RAG
        self._create_rag_chain()
        print("✅ Système RAG prêt !\n")
        
    def _load_existing_vectorstore(self):
        """Charge une base vectorielle existante."""
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=embeddings,
            collection_name="rag_documents"
        )
        
    def _create_vectorstore(self):
        """Crée une nouvelle base vectorielle."""
        # Chargement des documents
        print(f"📄 Chargement des PDFs depuis {self.data_path}...")
        loader = PyPDFDirectoryLoader(self.data_path)
        documents = loader.load()
        print(f"✅ {len(documents)} pages chargées")
        
        # Découpage
        print("✂️ Découpage en chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✅ {len(chunks)} chunks créés")
        
        # Embeddings
        print("🌐 Chargement du modèle d'embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Création de la base
        print("💾 Création de la base ChromaDB...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=self.chroma_path,
            collection_name="rag_documents"
        )
        print(f"✅ Base créée avec {len(chunks)} chunks")
        
    def _create_rag_chain(self):
        """Crée la chaîne RAG."""
        print("🤖 Initialisation du LLM...")
        
        # LLM
        llm = ChatOllama(
            model="qwen2.5:3b",
            temperature=0,
        )
        
        # Prompt
        system_prompt = """Tu es un assistant expert en intelligence artificielle.
        Réponds aux questions en utilisant UNIQUEMENT le contexte fourni.
        Si l'information n'est pas dans le contexte, dis-le clairement.
        
        Contexte :
        {context}
        
        Question : {question}
        
        Réponse :"""
        
        prompt = ChatPromptTemplate.from_template(system_prompt)
        
        # Chaîne
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        self.rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
        
    def ask(self, question: str):
        """Pose une question au système."""
        if not self.rag_chain:
            raise RuntimeError("Système non initialisé. Appelez setup() d'abord.")
        
        print(f"❓ Question : {question}\n")
        print("🔍 Recherche en cours...\n")
        
        response = self.rag_chain.invoke({"question": question})
        
        print("💬 Réponse :")
        print("=" * 80)
        print(response["answer"])
        print("=" * 80)
        
        print("\n📚 Sources :")
        for i, doc in enumerate(response["context"], 1):
            source = doc.metadata.get('source', 'N/A')
            page = doc.metadata.get('page', 'N/A')
            print(f"  {i}. {Path(source).name} (page {page})")
        
        return response


def main():
    """Fonction principale."""
    # Question par défaut
    default_question = "What is Retrieval Augmented Generation?"
    
    # Récupérer la question de la ligne de commande
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        print(f"ℹ️  Usage: python {sys.argv[0]} \"Your question here\"")
        print(f"ℹ️  Utilisation de la question par défaut...\n")
        question = default_question
    
    # Initialiser et utiliser le système
    try:
        rag = RAGSystem()
        rag.setup()
        rag.ask(question)
        
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        print("\n💡 Assurez-vous que :")
        print("  1. Ollama est installé et lancé (ollama serve)")
        print("  2. Le modèle est téléchargé (ollama pull qwen2.5:3b)")
        print("  3. Les dépendances sont installées (pip install -r requirements.txt)")
        print("  4. Le dossier data/arxiv contient des PDFs")
        sys.exit(1)


if __name__ == "__main__":
    main()

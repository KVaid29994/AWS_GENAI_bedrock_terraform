"""
=============================================================
🧠 HR Policy RAG (Retrieval-Augmented Generation) System
=============================================================

This script uses AWS Bedrock + LangChain to build a RAG system that:
1️⃣ Loads HR policy documents (PDF format)
2️⃣ Splits them into chunks for efficient embedding
3️⃣ Generates embeddings using Amazon Titan Embed model
4️⃣ Creates a FAISS vector store for retrieval
5️⃣ Queries the document using Amazon Titan Text model for LLM responses

Dependencies:
- boto3
- langchain
- langchain_community
- langchain_aws
- faiss

Author: Kashish Vaid
Date: October 2025
-------------------------------------------------------------
"""

import os
import boto3
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws.embeddings import BedrockEmbeddings  
from langchain_community.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain_aws import BedrockLLM as Bedrock


# =============================================================
# 📂 Function: Create HR Policy Vector Index
# =============================================================
def hr_index():
    """
    Loads HR policy document, splits text, creates embeddings,
    and stores them in a FAISS vector index.
    """
    # 1. Load the HR PDF document
    file_path = r"C:\Users\kanha\Desktop\AWS_genAI\RAG_HR_Doc\hr_policy_doc.pdf"
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 2. Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n', ' ', ''],
        chunk_size=100,
        chunk_overlap=5
    )
    split_docs = text_splitter.split_documents(documents)

    # 3. Initialize AWS Bedrock Embedding model
    aws_region = "us-east-1"
    bedrock = boto3.client(service_name="bedrock-runtime", region_name=aws_region)
    bedrock_embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        client=bedrock
    )

    # 4. Create FAISS index with LangChain’s VectorstoreIndexCreator
    data_index = VectorstoreIndexCreator(
        text_splitter=text_splitter,
        embedding=bedrock_embeddings,
        vectorstore_cls=FAISS
    )

    # 5. Generate and return the index
    db_index = data_index.from_loaders([loader])
    return db_index


# =============================================================
# 🤖 Function: Initialize Bedrock LLM
# =============================================================
def hr_llm():
    """
    Initializes Amazon Titan Text Express model as LLM.
    """
    bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

    llm = Bedrock(
        model_id="amazon.titan-text-express-v1",
        client=bedrock,
        model_kwargs={
            "temperature": 0.5,     
            "maxTokenCount": 300
        }
    )

    return llm


# =============================================================
# 💬 Function: Query RAG System
# =============================================================
def hr_rag_response(index, question):
    """
    Takes a question and retrieves an answer from the indexed HR policy data.
    """
    rag_llm = hr_llm()
    hr_rag_query = index.query(question=question, llm=rag_llm)
    return hr_rag_query


# =============================================================
# ✅ Example Usage (Uncomment for Testing)
# =============================================================
if __name__ == "__main__":
    index = hr_index()
    response = hr_rag_response(index, "What is the leave policy?")
    print(response)
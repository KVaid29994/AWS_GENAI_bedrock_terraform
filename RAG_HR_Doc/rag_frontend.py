"""
=============================================================
🧠 Streamlit Frontend for HR Policy RAG System
=============================================================
"""

import os
import streamlit as st
from rag_hr_backend import hr_index, hr_llm
from langchain_community.vectorstores import FAISS
from langchain_aws.embeddings import BedrockEmbeddings
import boto3
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# =============================================================
# 🎨 Page Configuration
# =============================================================
st.set_page_config(page_title="HR Policy Q&A Assistant", page_icon="🤖", layout="centered")

st.title("🧠 HR Policy Q&A Assistant")
st.markdown("Ask any question related to your HR policy document. Powered by **AWS Bedrock + LangChain RAG**.")

# =============================================================
# 🧩 Paths and Setup
# =============================================================
INDEX_DIR = "faiss_index"

aws_region = "us-east-1"
bedrock = boto3.client(service_name="bedrock-runtime", region_name=aws_region)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)

# =============================================================
# 🧠 Load or Create Index
# =============================================================
if "index" not in st.session_state:
    if os.path.exists(INDEX_DIR):
        with st.spinner("🔁 Loading saved HR Policy index..."):
            st.session_state.index = FAISS.load_local(
                INDEX_DIR, bedrock_embeddings, allow_dangerous_deserialization=True
            )
        st.success("✅ Index loaded from local storage!")
    else:
        st.session_state.index = None
        st.warning("⚠️ No saved index found. Please create it first.")


# =============================================================
# ⚙️ Index Creation Section
# =============================================================
st.header("📄 Step 1: Create or Load HR Policy Index")

if st.button("🔍 Create HR Policy Index"):
    with st.spinner("Processing and embedding HR policy document... ⏳"):
        try:
            index = hr_index()  # returns VectorstoreIndexWrapper
            os.makedirs(INDEX_DIR, exist_ok=True)
            index.vectorstore.save_local(INDEX_DIR)  # ✅ Save FAISS vectorstore
            st.session_state.index = index.vectorstore
            st.success("✅ HR Policy Index created and saved locally!")
        except Exception as e:
            st.error(f"❌ Failed to create index: {e}")

if st.session_state.index:
    st.info("✅ Index is ready. You can now ask questions below.")


# =============================================================
# 💬 Question-Answer Section
# =============================================================
st.header("💡 Step 2: Ask Your Question")

user_question = st.text_input("Enter your question:", placeholder="e.g., What is the annual leave policy?")

if st.button("🚀 Get Answer"):
    if not st.session_state.index:
        st.warning("⚠️ Please create or load the HR policy index first.")
    elif not user_question.strip():
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("Thinking... 🧠"):
            try:
                # 1️⃣ Create retriever from FAISS
                retriever = st.session_state.index.as_retriever(search_kwargs={"k": 3})

                # 2️⃣ Load Bedrock LLM
                llm = hr_llm()

                # 3️⃣ Define simple RAG QA chain
                prompt_template = """You are an HR assistant. Use the context below to answer the question accurately.

Context:
{context}

Question:
{question}

Answer:"""

                prompt = PromptTemplate(
                    input_variables=["context", "question"],
                    template=prompt_template,
                )

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs={"prompt": prompt}
                )

                response = qa_chain.run(user_question)

                st.success("✅ Answer:")
                st.markdown(f"**{response.strip()}**")

            except Exception as e:
                st.error(f"❌ Error during response generation: {e}")


# =============================================================
# 📘 Footer
# =============================================================
st.markdown("---")
st.caption("Built with ❤️ using Streamlit + AWS Bedrock + LangChain")

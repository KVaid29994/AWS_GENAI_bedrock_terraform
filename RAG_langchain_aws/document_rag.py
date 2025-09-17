from langchain_aws import BedrockLLM  as Bedrock ,BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
import boto3

aws_region = "us-east-1"
client = boto3.client(service_name = "bedrock-runtime", region_name = aws_region)

pdf_path = r"C:\Users\kanha\Desktop\aws learning\RAG_langchain_aws\islr_1.pdf"
loader = PyPDFLoader(pdf_path)
docs =loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

docs = splitter.split_documents(docs)
print(f"Loaded {len(docs)} chunks from {pdf_path}")

texts = [doc.page_content for doc in docs]


model = Bedrock(model_id = "amazon.titan-text-express-v1", client=client)

prompt = ChatPromptTemplate.from_messages([("system", "You are an helpful assisant. Answer the user question based on provided {context}",),  ("user", "{input}") ])

embedding_model = BedrockEmbeddings(model_id = "amazon.titan-embed-text-v2:0", client= client)

vector_databse = FAISS.from_texts(texts, embedding_model)
retriever = vector_databse.as_retriever(search_kwargs = {"k":3})
chain = prompt | model | StrOutputParser()

while True:
    user_input = input("Please enter your question or type'exit' to exit :")
    if user_input.lower() =="exit":
        break
    retrieved_docs = retriever.invoke(user_input)
    response_string = [rd.page_content for rd in retrieved_docs ]
    context_text = "\n\n".join(response_string)
    response = chain.invoke({"input": user_input, "context": context_text})
    print (response)
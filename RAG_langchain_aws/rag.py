from langchain_aws import BedrockLLM as Bedrock, BedrockEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
import boto3

my_data = [
    "Kashish is experienced in Python programming with a focus on backend development and API design.",
    "He is fair, smart and 5.10 feet tall.",
    "He is proficient in FastAPI, Pydantic, and JSON Schema for building modular and validated workflows.",
    "He is skilled in debugging complex systems, especially at the interface of code, data modeling, and API validation.",
    "He is fluent in concurrency patterns including threading, async/await, and task scheduling for scalable systems.",
    "He enjoys working on cloud solutions and is learning AWS services including EC2, Lambda, and Bedrock.",
    "He loves writing clean, testable code and mentoring juniors in best practices for software architecture.",
    "He is passionate about solving real-world problems with technology and experimenting with LLMs for automations."
]


question = "What technlogy does Kashish Know and what is his height"
aws_region = "us-east-1"

bedrock = boto3.client(region_name = aws_region, service_name = "bedrock-runtime")

model = Bedrock(model_id="amazon.titan-text-express-v1", client= bedrock)

bedrock_embeddings = BedrockEmbeddings(model_id = "amazon.titan-embed-text-v2:0", client= bedrock)

vector_store = FAISS.from_texts(my_data,bedrock_embeddings)

retriever = vector_store.as_retriever(search_kwargs ={"k":2})

template = ChatPromptTemplate.from_messages(
    [
        ("system","answer the user queston based on the provided content {context}",) ,("user", "{input}"),
    ]
)

chain = template.pipe(model)
# response = chain.invoke({"input": question,"context":results_string})
# print (response)

while True:
        question = input("\nAsk a question about Kashish (or type 'exit' to quit): ")
        if question.lower() =="exit":
              break
        results = retriever.invoke(question)
        results_string = [result.page_content for result in results]

        response = chain.invoke({"input":question, "context":results_string})
        print ("\n🤖 Answer:",response)

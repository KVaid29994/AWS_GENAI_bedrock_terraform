from langchain_aws import BedrockLLM as Bedrock
from langchain_core.prompts import PromptTemplate
import boto3

aws_region = "us-east-1"

bedrock = boto3.client(service_name = "bedrock-runtime", region_name = aws_region)

model = Bedrock(model_id = "amazon.titan-text-express-v1",client=bedrock, streaming= True)

# def invoke_model():
#     respone = model.invoke("What is the highest mountain in the world")
#     print (respone)

def first_chain():
    prompt = PromptTemplate.from_template( "write a short,compelling product description for :{product_name}"
                    )
    
    chain = prompt | model

    response = chain.invoke({"product_name":"bicyle"})
    print (response)

first_chain()
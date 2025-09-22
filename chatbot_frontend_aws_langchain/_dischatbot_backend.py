from langchain_aws import ChatBedrockConverse
from langchain.memory import ConversationSummaryBufferMemory
import boto3
import json
client = boto3.client(service_name = "bedrock-runtime", region_name = "us-east-1")


def demo_chatbot():
    demo_llm = ChatBedrockConverse(
        credentials_profile_name="default",
        model = "amazon.titan-text-express-v1",
        temperature=0.4,
        top_p=0.7,
        max_tokens=1000
    )
    return demo_llm


## function for conversation buffer memory (llm and max token limit)
def demo_memory():
    llm_data = demo_chatbot()
    memory = ConversationSummaryBufferMemory(llm = llm_data, max_token_limit=1000)
    return memory


### check invoke method

# user_message = "Explain in simple terms what Amazon VPC is and why it is useful"
# conversation = [
#     {"role": "user",
#      "content" : [{"text":user_message}]
#     }
# ]

# response = demo_chatbot(conversation=conversation)
# print (response)


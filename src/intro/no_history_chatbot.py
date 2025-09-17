import boto3
import json
from botocore.exceptions import ClientError

# Create a Bedrock Runtime client in the AWS Region of your choice.
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Set the model ID, e.g., Titan Text Premier.
model_id = "amazon.titan-text-express-v1"

def get_configurration(prompt:str):
    return json.dumps({
    "inputText":prompt,
    "textGenerationConfig": {
        "maxTokenCount": 512,
        "temperature": 0.5, "topP":1,
    },
})


print ("Bot : Hello! I am a chabot! Ask me a question")

while True:
    user_input = input("user: ")
    if user_input.lower() =="exit":
        break
    response = client.invoke_model(body = get_configurration(user_input), modelId=model_id)
    response_body = json.loads(response.get("body").read())
    print (response_body["results"][0]["outputText"])

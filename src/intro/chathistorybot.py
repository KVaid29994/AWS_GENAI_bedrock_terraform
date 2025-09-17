import boto3
import json
# from botocore.exceptions import ClientError

# Create a Bedrock Runtime client in the AWS Region of your choice.
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Set the model ID, e.g., Titan Text Premier.
model_id = "amazon.titan-text-express-v1"

history = []
def get_history():
    return    "\n".join(history)


def get_configurration(prompt:str):
    return json.dumps({
    "inputText":get_history(),
    "textGenerationConfig": {
        "maxTokenCount": 512,
        "temperature": 0.5, "topP":1,
    },
})


print ("Bot : Hello! I am a chabot! Ask me a question")

while True:
    user_input = input("user: ")
    history.append("user: " +user_input)
    if user_input.lower() =="exit":
        break
    response = client.invoke_model(body = get_configurration(user_input), modelId=model_id)
    response_body = json.loads(response.get("body").read())
    output_text = response_body["results"][0]["outputText"].strip()
    print(output_text)
    history.append(output_text)

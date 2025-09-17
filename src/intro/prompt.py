import boto3
import json


client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Set the model ID, e.g., Titan Text Premier.
model_id = "amazon.titan-text-express-v1"

prompt = '''

Sample Conversation

User: Hey, I’m planning a short trip to Manali. Can you suggest some must-visit places?
Assistant: Absolutely! You should check out Solang Valley for adventure sports, Hidimba Devi Temple for culture, and Rohtang Pass if it’s open during your visit.
User: Sounds great. What about food—any local dishes I shouldn’t miss?
Assistant: Yes! Try sidu (a type of bread), trout fish, and the Himachali thali. Street vendors and local cafés in Old Manali are perfect for these.
User: Nice, and can you also tell me the best time of year to go?
Assistant: March to June for pleasant weather, or December to February if you’re looking for snow adventures.

Summarize the conversation between a user and an assistant. Capture the main purpose of the discussion, the recommendations given, and the context (such as travel, food, timing, etc.). The summary should be concise, clear, and highlight the key takeaways without unnecessary detail. Present it in 3–4 sentences or max 50 words."

'''

titan_config = {
    "inputText":prompt,
    "textGenerationConfig": {
        "maxTokenCount": 256,
        "temperature": 0.5, "topP":1,
    },
}

request = json.dumps(titan_config)
print(request)
response = client.invoke_model(modelId=model_id, body=request)


model_response = json.loads(response["body"].read())

response_text = model_response["results"][0]["outputText"]
print(response_text)
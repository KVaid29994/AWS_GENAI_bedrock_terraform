import boto3
import json

aws_region = "us-east-1"

client = boto3.client(service_name = "bedrock-runtime", region_name = aws_region)

def lambda_handler(event,context):
    input_prompt = event['prompt']
    
    response = client.invoke_model(
        body=json.dumps({
          "prompt" : input_prompt 
        },
        contentType ='pplication/json',
        modelId = "amazon.titan-text-express-v1"
    ))

    response_byte = response["body"].read()
    response_string = json.loads(response_byte)
    response_final = response_string['generations'][0]['text']


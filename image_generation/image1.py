import boto3
import json
import datetime
import base64

aws_region = "us-east-1"

client = boto3.client(service_name = "bedrock-runtime",region_name = aws_region)

s3client = boto3.client(service_name = "s3", region_name = aws_region)

def lambda_handler(event, context):
    input_prompt = event['prompt']
    print ("the input from user",input_prompt)
    native_request = {
    "text_prompts": [{"text": input_prompt}],
    "style_preset": "photographic",
    "seed":0,
    "cfg_scale": 10,
    "steps": 10,}
    request = json.dumps(native_request)
    response_bedrock = client.invoke_model(
        modelId = "stability.stable-diffusion-xl-v1",
        body  = request
    )
    model_response = json.loads(response_bedrock["body"].read())
    base64_image_data = model_response["artifacts"][0]["base64"]
    image_data = base64.b64decode(base64_image_data)

    poster_name = "posterName" + datetime.datetime.today().strftime("%Y-%M-%D-%M-%S")
    
    response_s3 = s3client.put_object(
        Bucket ="kvmovieposterdesign",
        Body = image_data,
        Key = poster_name
    )
    presigned_url = s3client.generate_presigned_url(
        ClientMethod ="get_object",
        Params = {"Bucket":"kvmovieposterdesign",
                  "Key" :"poster_name"
                  }, 
        ExpiresIn=3600
    )
    return {
        'statusCode': 200,
        'body': presigned_url
    }




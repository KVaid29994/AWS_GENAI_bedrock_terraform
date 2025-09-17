# Generate and print an embedding with Amazon Titan Text Embeddings V2 (Custom Size)
import boto3
import json

# Create a Bedrock Runtime client in the AWS Region of your choice.
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Set the model ID, e.g., Titan Text Embeddings V2.
model_id = "amazon.titan-embed-text-v2:0"

# The text to convert to an embedding.
input_text = "Please recommend books with a theme similar to the movie 'Inception'."

# Create the request for the model with custom dimensions
# Supported dimensions: 256, 512, 1024 (default)
embedding_dimensions = 512  # Change this to 256, 512, or 1024

native_request = {
    "inputText": input_text,
    "dimensions": embedding_dimensions
}

# Convert the native request to JSON.
request = json.dumps(native_request)

try:
    # Invoke the model with the request.
    response = client.invoke_model(modelId=model_id, body=request)
    
    # Decode the model's native response body.
    model_response = json.loads(response["body"].read())
    
    # Extract and print the generated embedding and the input text token count.
    embedding = model_response["embedding"]
    input_token_count = model_response["inputTextTokenCount"]
    
    print("\nYour input:")
    print(input_text)
    print(f"Number of input tokens: {input_token_count}")
    print(f"Requested embedding dimensions: {embedding_dimensions}")
    print(f"Actual size of the generated embedding: {len(embedding)}")
    print("Embedding (first 10 values):")
    print(embedding[:10])  # Print only first 10 values for readability
    
except Exception as e:
    print(f"Error: {e}")

# Example: Compare different embedding sizes
print("\n" + "="*50)
print("COMPARING DIFFERENT EMBEDDING SIZES")
print("="*50)

for dims in [256, 512, 1024]:
    try:
        request_data = {
            "inputText": input_text,
            "dimensions": dims
        }
        
        response = client.invoke_model(
            modelId=model_id, 
            body=json.dumps(request_data)
        )
        
        result = json.loads(response["body"].read())
        embedding_size = len(result["embedding"])
        
        print(f"Requested: {dims} dimensions → Actual: {embedding_size} dimensions")
        
    except Exception as e:
        print(f"Error with {dims} dimensions: {e}")
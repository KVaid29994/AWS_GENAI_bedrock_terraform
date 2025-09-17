import boto3
import pprint

bedrock = boto3.client(service_name = "bedrock",region_name = "us-east-1")

# models = bedrock.list_foundation_models()

# print (models)

pp = pprint.PrettyPrinter(depth=4)
def list_foundation_models():
    models = bedrock.list_foundation_models()
    for model in models["modelSummaries"]:
        pp.pprint(models)
        pp.pprint("-------")


# list_foundation_models()
def get_foundation_model(modelIdentifier):
    model = bedrock.get_foundation_model(modelIdentifier=modelIdentifier)
    pp.pprint(model)



get_foundation_model('meta.llama4-scout-17b-instruct-v1:0')


# as seen here: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-invoke.html#inference-examples-stream
import boto3
import json

brt = boto3.client(service_name='bedrock-runtime')

body = json.dumps({
    'prompt': '\n\nHuman: write an essay for living on mars in 1000 words\n\nAssistant:',
    'max_tokens_to_sample': 4000
})
                   
response = brt.invoke_model_with_response_stream(
    modelId='anthropic.claude-v2', 
    body=body
)
    
stream = response.get('body')
if stream:
    for event in stream:
        chunk = event.get('chunk')
        if chunk:
            print(json.loads(chunk.get('bytes').decode()))

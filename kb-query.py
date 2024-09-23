import boto3
import json
import re

# Initialize Bedrock clients
bedrock_agent = boto3.client('bedrock-agent-runtime')
bedrock_runtime = boto3.client('bedrock-runtime')

def query_knowledge_base(query, knowledge_base_id):
    response = bedrock_agent.retrieve(
        knowledgeBaseId=knowledge_base_id,
        retrievalQuery={
            'text': query
        },
        retrievalConfiguration={
            'vectorSearchConfiguration': {
                'numberOfResults': 3
            }
        }
    )
    return response['retrievalResults']

def get_formatted_response(model_name, response):
    if re.search("claude", model_name, re.IGNORECASE):
        response_body = json.loads(response.get('body').read())["completion"]
    elif re.search("titan", model_name, re.IGNORECASE):
        response_body = json.loads(response.get('body').read())["results"][0]["outputText"]
    elif re.search("llama2", model_name, re.IGNORECASE):
        response_body = json.loads(response.get('body').read())["generation"]
    else:
        raise ValueError("Unsupported model type")
        
    return response_body

def get_formatted_prompt(model_name, prompt, temperature, max_tokens, top_p):
    templates = {
        "claude-3": {},          
        "claude-instant": {
            "prompt": "\n\nHuman:{}\n\nAssistant:".format(prompt),
            "max_tokens_to_sample": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },        
        "claude-v2": {
            "prompt": "\n\nHuman:{}\n\nAssistant:".format(prompt),
            "max_tokens_to_sample": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
        "titan": {   
            "inputText": "\n\nHuman:{}\n\nAssistant:".format(prompt),
            "textGenerationConfig" : { 
                "maxTokenCount": max_tokens,
                "stopSequences": [],
                "temperature": temperature,  
                "topP": top_p
            }
        },
        "llama2": {
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_gen_len": max_tokens
        }
    }

    for model_key in templates:
        if model_key in model_name:
            return templates[model_key]

    raise ValueError("No matching template found for model: {}".format(model_name))

def make_bedrock_call(model, prompt):
    try:
        temperature = 0
        formatted_prompt = get_formatted_prompt(model, prompt, temperature, 300, 0.9)
    
        body = json.dumps(formatted_prompt)
        response = bedrock_runtime.invoke_model(body=body, modelId=model, accept='application/json', contentType='application/json')
    
        response_body = get_formatted_response(model, response)
    except Exception as e:
        return {
            'status_code': 500,
            'body': json.dumps('Error calling AWS Bedrock: ' + str(e))
        }

    return {
        'status_code': 200,
        'body': response_body
    }

def main():
    knowledge_base_id = '<your_knowledge_base_id_here>'
    model_id = "anthropic.claude-v2"
    user_query = "What is a good club if I like video games?"
    
    print("Querying knowledge base...")
    kb_results = query_knowledge_base(user_query, knowledge_base_id)
    
    context = "\n".join([result['content']['text'] for result in kb_results if 'content' in result and 'text' in result['content']])
    
    prompt = f"""Context information:
    {context}

    Based on the above information, please provide a recommendation or answer to the following question:
    {user_query}

    IMPORTANT: provide only recommendations that reflect club information in this query. Strive for 100% factual accuracy.
    """

    print(prompt)
    
    print("Generating response using LLM...")
    response = make_bedrock_call(model_id, prompt)

    if response["status_code"] == 200:
        print("Response:")
        print(response["body"])
    else:
        print("Error:", response["body"])

if __name__ == "__main__":
    main()


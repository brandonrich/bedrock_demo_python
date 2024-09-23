import boto3
import json

# Initialize Bedrock client
bedrock = boto3.client('bedrock-agent-runtime')

def query_knowledge_base(query, knowledge_base_id):
    response = bedrock.retrieve(
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

def generate_with_llm(prompt, model_id):
    body = json.dumps({
        "prompt": prompt,
        "max_tokens_to_sample": 300,
        "temperature": 0.7,
        "top_p": 0.9,
    })
    
    response = bedrock.invoke_model(
        modelId=model_id,
        body=body
    )
    
    return json.loads(response['body'].read())['completion']

def main():
    knowledge_base_id = '<your_knowledge_base_id_here>'
    model_id = 'anthropic.claude-v2'  # or any other supported model
    user_query = "What is a good club if I like video games?"
    
    # Query knowledge base
    print("querying kb")
    kb_results = query_knowledge_base(user_query, knowledge_base_id)
    print("results:" )
    print(kb_results)
    
    # Construct prompt with retrieved information
    #context = "\n".join([result['content'] for result in kb_results])
    #prompt = f"""Context information:
    # {context}

if __name__ == "__main__":
    main()

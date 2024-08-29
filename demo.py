import boto3
import json
import re

# ********************************************************************************************************************
def get_formatted_response(model_name, response):
    if re.search("claude", model_name, re.IGNORECASE):
        print("get_formatted_response: claude")
        response_body = json.loads(response.get('body').read())["completion"]
    elif re.search("titan", model_name, re.IGNORECASE):
        print("get_formatted_response: titan")
        response_body = json.loads(response.get('body').read())["results"][0]["outputText"]
    elif re.search("llama2", model_name, re.IGNORECASE):
        response_body = json.loads(response.get('body').read())["generation"]
    else:
        return "No match found" # this should throw an exception later
        
    return response_body


def get_formatted_prompt(model_name, prompt, temperature, max_tokens, top_p):
    # Define templates for different models
    templates = {
        "claude-3": {
          
        },          
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

    # Find the matching template
    for model_key in templates:
        if model_key in model_name:
            # Fill in the template with the provided parameters
            return templates[model_key]

    # If no matching template is found
    return None

def make_bedrock_call(model, prompt):
   # Make an inference call to AWS Bedrock
    # Note: Replace 'bedrock_client_function' with the actual function to call AWS Bedrock
    try:
        bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
    
        temperature = 0
        formatted_prompt = get_formatted_prompt(model, prompt, temperature, 300, 0.9)
    
        body = json.dumps(formatted_prompt)
        modelId = model 
        accept = 'application/json'
        contentType = 'application/json'
    
        print("calling bedrock")
        #print(formatted_prompt)
        response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    
        #response_body = json.loads(response.get('body').read())
        response_body = get_formatted_response(model, response)
    except Exception as e:
        return {
            'status_code': 500,
            'body': json.dumps('Error calling AWS Bedrock: ' + str(e))
        }

    # Return the response
    return {
        'status_code': 200,
        'body': response_body
    }



def main():
    try:
        # After establishing a command-line session using the credentials copied from
        # awsconsole.nd.edu, Bedrock foundational models and their IDs can be listed from 
        # the command line with this command: 
        # aws bedrock list-foundation-models --query 'modelSummaries[*].[modelId,modelName]' --output table
        
        # NOTE that each foundation model has slightly different input and output format expectations.  
        # For Claude 2x, Llama 2, and Titan, these are handled by the functions "get_formatted_prompt" 
        # and "get_formatted_response" in this file.
        # Details about the Claude 3 input format are here: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
        
        # This file has code that talks directly to language models through Bedrock.  If you want
        # a library that abstracts this even further (so you don't have to worry about specific input formats),
        # see LangChain: https://www.langchain.com/ 
        prompt = "Why is the sky blue?"
        model_id = "anthropic.claude-v2:1"
        print ("Model ID: " + model_id)
        print ("Prompt: " + prompt)
        response = make_bedrock_call(model_id, prompt)

        if ( response["status_code"] != 200 ): 
            raise Exception("Non-200 response code: " + response["body"])   
        
        print("Got response: ", response["body"])
    except Exception as e:
        return e

if __name__ == "__main__":
    main()

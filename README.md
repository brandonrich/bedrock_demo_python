# Bedrock Demo Python
Shows how to call AWS Bedrock from Python.


## Running the script
To run this script, first establish your AWS account credentials on the command line.  Visit awsconsole.nd.edu, then click the arrow key next to your account name:

<img width="981" alt="image" src="https://github.com/user-attachments/assets/62449742-52cc-47db-a138-6990047ba556">

From here, click "Copy" under "Option 1: Set AWS environment variables"

<img width="808" alt="image" src="https://github.com/user-attachments/assets/a59729b4-b6e9-4c1b-9774-e703c9f3a359">


Paste this into a terminal session and hit Enter.  You can now run this script:

`> python3 demo.py`

Response should be similar to this:

```
Model ID: anthropic.claude-v2:1
Prompt: Why is the sky blue?
calling bedrock
get_formatted_response: claude
Got response:   The sky appears blue because of how air in Earth's atmosphere interacts with sunlight. As sunlight passes through the atmosphere, some of the colors with shorter wavelengths, like blue and violet light, are scattered more by air molecules. This scattered blue light is what makes the sky look blue from the ground.
```


## Understanding the script

The `main` function calls a function called `make_bedrock_call`.  This is a helper function that wraps calls to the Bedrock service through `boto3`, the Python SDK for AWS.  The sequence is simple:
  * format the prompt as required by the chosen model
  * pass the prompt to the model
  * extract the response using the format returned by the model.

Each foundation model has slightly different input and output format expectations.  For Claude 2x, Llama 2, and Titan, these are handled by the helper functions `get_formatted_prompt`
and `get_formatted_response` in this file. To use newer models, you will need to make sure you implement the right input / output format support!

For instance, details about the Claude 3 input format are here: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html

### Understanding Model IDs

Model IDs can be found here: https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html

They can also be fetched programmatically. After establishing a command-line session using the credentials copied from
awsconsole.nd.edu, Bedrock foundational models and their IDs can be listed from 
the command line with this command: 

`aws bedrock list-foundation-models --query 'modelSummaries[*].[modelId,modelName]' --output table`

This file has code that talks directly to language models through Bedrock.  If you want
a library that abstracts this even further (so you don't have to worry about specific input formats),
see LangChain: https://www.langchain.com/ 



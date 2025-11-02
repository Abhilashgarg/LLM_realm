## for openai - connection setup with llm model

# from langchain_openai import ChatOpenAI
# import os 
# import httpx
# client=httpx.Client(verify=False)
# llm=ChatOpenAI(
#     base_url="https://xxxxxx.xxxx.in"
#     model="azure_ai/xxxxxxxx-mass-DeepSeek-V3_0324",
#     api_key="XXXXXXXXXXXXXX",
#     http_client=client)

####  update ####

from huggingface_hub import InferenceClient

HF_TOKEN = "XXXXXXX"

client = InferenceClient(
    model="meta-llama/Llama-3.3-70B-Instruct",
    token=HF_TOKEN
)

# Conversational call
response = client.chat_completion(
    messages=[
        {"role": "user", "content": "Hi"}
    ],
    max_tokens=100
)

# Print model response
print(response.choices[0].message["content"])



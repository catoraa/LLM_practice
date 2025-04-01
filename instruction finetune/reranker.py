import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset
from openai import OpenAI

prompt_style = """
Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request.
### Instruction:
Given a series of sentences B, analyze the topic relevance to sentence A , rank in descending order.
Return a complete sorted array of ordinal numbers.
No further explanation or justification is required.
### Question:
A:["今天天气真好。"]
B:["我想吃饭了。","今天出太阳了。","今天天气不好。","今天天气真好。"]
### Response:
{}
"""
client = OpenAI(api_key='lm-studio',
                base_url='http://192.168.123.82:1234/v1')
dataset = load_dataset("glue", "sst2", split="validation[:100]")
model_id = "gemma-2-9b-it"

response = client.chat.completions.create(
    messages=[{'role': 'user', 'content': prompt_style.format("")}],
    model=model_id
)
print(response.choices[0].message.content)

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset
from openai import OpenAI

prompt_style = """
Below is an instruction that describes a task,paired with an input that provides further context. 
Write a response that appropriately completes the request.
### Instruction:
Analyze the given text from an online review and determine the sentiment polarity. 
Return a single number of either 0 and 1, with 0 being negative and 1 being the positive sentiment.
No further explanation or justification is required.
### Question:
{}
### Response:
{}
"""
client = OpenAI(api_key='lm-studio',
                base_url='http://192.168.123.82:1234/v1')
dataset = load_dataset("glue", "sst2", split="train[-20:]")
model_id = "qwq-32b"

correct = 0
for i, item in enumerate(dataset):
    input_text = item['sentence']
    response = client.chat.completions.create(
        messages=[{'role': 'user', 'content': prompt_style.format(input_text, "")}],
        model=model_id
    )
    print(response.choices[0].message.content)
    if str(response.choices[0].message.content[-1]) == str(item['label']):
        correct += 1
        print(f"true,进度{i + 1}/{len(dataset)},正确率{correct / (i + 1) * 100:.2f}%")
    else:
        print(f"false,进度{i + 1}/{len(dataset)},正确率{correct / (i + 1) * 100:.2f}%")

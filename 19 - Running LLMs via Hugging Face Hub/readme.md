Hugging page is a website that is used to store and share AI models. 
In here you can run you models or any models

It's an onestop solution for all the information related to the models.

2. Securing and creating the hugging face

It's an simple signup and login then you can see all the models 

3. Installing and Using Hugging face CLI Tools

for this we need to install the huggingface-cli

to install you need to use
pip install -U "huggingface_hub"
then login
hf auth login

4. Modeling Downloading & Execution From HF Hub
We need to install transformer

5. then wrtite the code
from transformers import pipeline
import torch

pipe = pipeline(
    "image-text-to-text",
    model="google/gemma-3-4b-it",
    device="cuda",
    torch_dtype=torch.bfloat16
)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    }
]

print(output[0]["generated_text"][-1]["content"])


from here what it will d it will download ther modal and then you can use it
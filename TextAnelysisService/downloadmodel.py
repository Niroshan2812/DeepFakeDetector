from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import os

def download_gp2_locally():
    model_name = "gpt2"
    local_dir = "./local_nlp_model"

    tokenizer = GPT2LMHeadModel.from_pretrained(model_name)
    model =GPT2LMHeadModel.from_pretrained(model_name)

    os.makedirs(local_dir, exist_ok=True)
    tokenizer.save_pretrained(local_dir)
    model.save_pretrained(local_dir)

    print("Downloded")

if __name__ =="__main__":
    download_gp2_locally()
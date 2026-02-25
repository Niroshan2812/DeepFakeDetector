from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import numpy as np
import re

app=FastAPI(title="TEXT and NLP liveness Service")

print("Loading loal model")
local_model_path = "./local_nlp_model"
tokenizer=GPT2TokenizerFast.from_pretrained(local_model_path)
model=GPT2LMHeadModel.from_pretrained(local_model_path)
print("model loaded ")

class TextPayload(BaseModel):
    text:str

def sanitize_input(text:str)->str:
    # remove zero-weidth craracter invisible formatting 
    clean_text = re.sub(r'[\u200B-\u200D\uFEFF]','',text)
    #Remove multiple space and newlines
    clean_text=re.sub(r'\s',' ',clean_text).strip()
    return clean_text

def calculate_burstuness(text:str)->float:
    """
    split the sentized text into individual sentence using punctuation marks
    then calculate number of  words (lengths) in each sentence
    then get std of those lengths 
    Fact --> human - high brustuness, while ai is low 
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip()
                 for s in sentences
                    if len(s.strip())>0
                 ]
    # cannt calculate variance on a single sentence 
    if len(sentences) <=1:
        return 0.0
    
    lenngths =[len(s.split())
               for s in sentences]
    return float(np.std(lenngths))

def calculate_perplexity(text:str)->float:
    """
    convert text into tokenID using gpt-2 tokenizer
    model terutn the loss --> how surprised it is by the nexr word srquence 
    then calculate perplexity by taking the exponential of that loss, 
    low perplexity --> high predictable 
    """

    inputs = tokenizer(text, return_tensors="pt")

    if inputs["input_ids"].shape[1]<2:
        raise ValueError("Payload generated less than 2 tokens. Cannot calculate next-word perplexity.")
    with torch.no_grad():
        outputs =model(**inputs,labels=inputs["input_ids"])
        loss=outputs.loss
        perplexity=torch.exp(loss)
    return float (perplexity)

@app.post("/api/v1/text/analyze")
async def analyze_text(payload:TextPayload):
    clean_text = sanitize_input(payload.text)

    #min length validation 
    word_count = len(clean_text.split())
    if word_count < 30:
        return {"status":"failed", "reason":f"Text is too short {word_count} min is 30"}
    
    try:
        perplexity = calculate_perplexity(clean_text)
        burstiness = calculate_burstuness(clean_text)

        # Scoring 
        confidence_score = 1.0

        #penalize low perplexity 
        if perplexity < 40:
            confidence_score -=0.4
        elif perplexity <60:
            confidence_score -=0.2
        #penalize low Burstiness 
        if burstiness < 3.0:
            confidence_score -=0.3
        
        final_score = max(0.01,min(confidence_score,1.0))

        return {
            "status": "passed",
            "human_probabillity_score": round(final_score,3),
            "metrics":{
                "word_count":word_count,
                "perplexity":round(perplexity,2),
                "brustiness_variance": round(burstiness, 2)
            }
        }
    except ValueError as ve:
        return {"status": "failed", "reason":str(ve)}
    except Exception as e:
        return {"status":"failed", "reason": str(e)}
    
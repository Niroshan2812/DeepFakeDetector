from fastapi import FastAPI, UploadFile, File, HTTPException 
import librosa
import numpy as np
import tempfile 
import os 
from transformers import pipeline
import subprocess

app = FastAPI(title ="Audio forensic Service ")

# Load the AI model into memory when the server starts 
print("Loading AI model")
# Save model on local file system 
# manually download it then use inside 
# Url -> https://huggingface.co/mo-thecreator/Deepfake-audio-detection/tree/main
# Make filename -local_audio_model then include - config.json, model.safetensors, preprosessor_config.json
local_model_pat = "./local_audio_model"

try:
    subprocess.run(["ffmpeg","-version"],check= True, capture_output=True)
    print ("FFmpeg is have")
except FileNotFoundError:
    print("FFpeg not inlude")
    exit(1)

audio_classifier = pipeline(
    task = "audio-classification",
    model= local_model_pat,
    feature_extractor=local_model_pat,
    device = "cpu"
)
print("Model is loaded ")

def extract_phase_irregularity(y):
    # Calculate STFT
    D = librosa.stft(y)

    # Extract phase
    _, phase = librosa.magphase(D)
    phase_angle = np.angle(phase)

    #Unwrap and calculate instantaneous freq..
    unwrapped_phase = np.unwrap(phase_angle, axis=1)
    phase_derivative = np.diff(unwrapped_phase, axis=1)

    #Calulate variance as bies inrregulatiy matic 
    phase_variance = np.var(phase_derivative)
    return phase_variance

def anelyze_audio_forensics(audio_path):
    # Mathemetical DSP analysis
    try:
        y,sr=librosa.load(audio_path, sr=22050)

        if len(y)<sr * 1:
            return {"status": "failed", "reason": "Audio payload too short for NLP/Forensic analysis."}

        # 2. Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # Analyze high freq varinace -- looking for missing breath 
        high_mfcc_variance = np.var(mfccs[4:])
        # Anelyse ohase irregukaties 
        phase_variance = extract_phase_irregularity(y)

        # AI model analysis
        ai_predictions = audio_classifier (audio_path)
        ai_human_score = 0.0

        for pred in ai_predictions:
            if pred ['label'].lower() in ['bonafide', 'real', 'human', 'label_0']:
                ai_human_score = pred["score"]
                break
        
        # confidence score 
        confidencr_score = ai_human_score


        if high_mfcc_variance < 150:
            confidencr_score -= 0.2
        if phase_variance > 2.5:
            confidencr_score -=0.2
        
        final_score = max(0.01,min(confidencr_score, 1.0))

        return {
            "status": "passed",
            "human_probability_score": round(final_score,3),
            "engine_used": "Hybrid (Wav2Vec2 + DSP Math)",
            "metrics":{
                "ai_base_score": round(float(ai_human_score), 3),
                "high_mfcc_variance": round(float(high_mfcc_variance),2),
                "phase_variance":round(float(phase_variance),2)
            }
        }
    except Exception as e:
        return {"status" : "failed", "reason": str(e)}
    
@app.post("/api/v1/audio/verify")
async def verify_audio(audio:UploadFile=File(...)):
    with tempfile.NamedTemporaryFile(delete=False,suffix=".wav")as tmp:
        tmp.write(await audio.read())
        tmp_path=tmp.name

    try : 
        result = anelyze_audio_forensics(tmp_path)
    finally:
        os.remove(tmp_path)
    return result
from fastapi import FastAPI, UploadFile, File, HTTPException 
import librosa
import numpy as np
import tempfile 
import os 

app = FastAPI(title ="Audio forensic Service ")

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

        confidencr_score = 1.0

        if high_mfcc_variance < 150:
            confidencr_score -= 0.4
        if phase_variance > 2.5:
        
            confidencr_score -=0.3
        
        final_score = max(0.01,min(confidencr_score, 1.0))

        return {
            "status": "passed",
            "human_probability_score": round(final_score,3),
            "metrics":{
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
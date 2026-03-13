from fastapi import FastAPI, UploadFile, File
import math
import os

app=FastAPI(title="Reverse enginaring service")

# A dictionary of known hex   signatures to find the compilerSoftware
SIGNATURES={
    b"Lavf": "FFmpeg (Highly common in automated Python AI scripts)",
    b"Adobe": "Adobe Creative Cloud (Human edited)",
    b"GIMP": "GIMP Image Editor (Human edited)",
    b"Isom": "Standard MP4 Base Media",
    b"StableDiffusion": "Stable Diffusion Metadata",
}

def calculate_entropy(chunk:bytes)-> float:
    #calculate shannon entropy to detect encription or DRM
    """
        The main purpose is - use entropy consept 
        for predict how much unpredictable or How much 
        random data is, 
        So low entropy mean predictable. while high mean very random 
    """
    if not chunk:
        return 
    entropy =0
    for x in range(256):
        p_x = float(chunk.count(x))/len(chunk)
        if p_x>0:
            entropy += - p_x * math.log(p_x,2)
    return entropy

@app.post("/api/v1/reverseEngineering/analyze")
async def anelyze_file(file: UploadFile=File(...)):
    chunk_size =8192
    found_signature = set()
    is_first_chunk=True
    header_entropy=0.0

    try: 
        while True:
            #Read exactly 8kb at a time, 
            chunk=await file.read(chunk_size)
            if not chunk:
                break

            # DRM and encription check 
            if is_first_chunk:
                header_entropy = calculate_entropy(chunk)

                # pure random encription data approches an entripy of 8.0 
                """
                    Each bit can be 0 or 1 
                    so a byte can have 256 possible values 
                    if every possible byte appears equely likely then rntopy is 
                    8 bits per byte -> so that is the highest randomness can get for 
                    byte-baced data (try to enhance with -- PQC approch  )
                """
                if header_entropy > 7.9:
                    return {
                        "status": "failed",
                        "reason": "High entropy detected. file is havily encripted or wrapped in stratic DRM"
                    }
                is_first_chunk = False

            for sig, name in SIGNATURES.items():
                if sig in chunk:
                    found_signature.add(name)
    except Exception as e:
        return {"status":"failed","reason":f"Stram reading error: {str(e)}"}
    
    # Scoring logic 
    score = 1.0

    # penalize programmtic assembly tools 
    if "FFmpeg (Highly common in automated Python AI scripts)" in found_signature:
        score -= 0.3
    if "Stable Diffusion Metadata" in found_signature:
        score -= 0.9
    return {
        "statu":"passed",
        "human_probability_score": max(0.01,score),
        "metrics":{
            "header_entropy":round(header_entropy,2),
            "detected_fingerprints":list (found_signature) 
        }
    }

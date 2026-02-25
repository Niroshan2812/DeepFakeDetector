from fastapi import FastAPI, UploadFile, File
from transformers import pipeline
from PIL import Image
import cv2
import numpy as np
import tempfile
import os 

app =FastAPI(title="Vision Artifact service")

"""
def download_vision_model ():
    model_name= "umm-maybe/AI-image-detector"
    local_dir = "./local_vision_model"

    print(f"Downloading {model_name} from huggingface ")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)

    os.makedirs(local_dir, exist_ok=True)
    processor.save_pretrained(local_dir)
    model.save_pretrained(local_dir)
    
    print(f"Success! Model permanently saved to: {local_dir}")
"""

print("Loading local AI")
local_model_path = "./local_vision_model"

image_classifier_pipline = pipeline(
    "image-classification",
    model=local_model_path,
    device="cpu"
)
print("Local model loaded")

def analyse_version_artifacts(image_path:str)->dict:
    try:
        # AI model prediction 
        
        ai_result = image_classifier_pipline(image_path)

        ai_top_lable = ai_result[0]['label']
        ai_confidence=ai_result[0]['score']

        ai_human_score = ai_confidence if ai_top_lable.lower() in ['real', 'human'] else (1.0-ai_confidence)

        # Frequency anelysis (cv)
        #load Image
        img= cv2.imread(image_path)
        if img is None:
            return {"status":"failed","reason":"invalid or curruped image file"}
        
        # OOM prevention --- downscale n>=4k
        max_dim=1024
        h,w=img.shape[:2]
        if h>max_dim or w>max_dim:
            scaling_facor = max_dim/float(max(h,w))
            img=cv2.resize(img,None, fx=scaling_facor, fy=scaling_facor, interpolation = cv2.INTER_AREA)

        # convert to grayscale
        gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # adversarial attach prevention 
        smoothed =cv2.GaussianBlur(gray,(3,3),0)

        #Fast fourier transform 
        f_transform = np.fft.fft2(smoothed)
        f_shift=np.fft.fftshift(f_transform)

        #magnitute spectrum calculation 
        magnitude_spectrum = 20*np.log(np.abs(f_shift)+1e-8)

        #frequency analysis math 
        rows, cols = gray.shape
        crow,ccol = rows//2,cols//2

        #mark out the low frequenct center 
        mask=np.ones((rows, cols), np.uint8)
        r=50
        center = [crow, ccol]
        x,y=np.ogrid[:rows,:cols]
        mask_area = (x - center[0])**2+(y-center[1])**2 <= r*r
        mask[mask_area]=0
        

        # Apply marsks to get only high frequencies 
        high_freq_specturm = magnitude_spectrum*mask

        #calculate variance of the high frequencies
        # High freq --> AI tiling 
        high_freq_variance = np.var(high_freq_specturm[mask ==1])

        # Scoring Logic 
        confidence_score = 1.0

        if high_freq_variance > 3000:
            confidence_score -= 0.6
        elif high_freq_variance > 1500:
            confidence_score -= 0.3

        fft_final_score = max(0.01,min(confidence_score, 1.0))

        # Combine both approch 
        # fOR testing i only include ai model prediction 
        blended_score = (ai_human_score *1) + (fft_final_score*0.0)

        return {
            "status":"passed",
            "human_pron_score":round(blended_score,3),
            "metrics":{
                "ai_model_score": round(float(ai_human_score),3),
                "ai_model_label":ai_top_lable,
                "fft_math_score":round(float(fft_final_score),3),
                "high_frequenct_variance":round(float(high_freq_variance),2),
                "adversarial_defense_applied":"Gaussian Blur (3x3)"

            }
        }
    except Exception as e:
        return {"status":"failed", "reason":str(e)}

@app.post("/api/v1/vision.analyze")
async def analyze_image(file:UploadFile=File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(await file.read())
        tmp_path=tmp.name

    try :
        result= analyse_version_artifacts(tmp_path)
    finally:
        os.remove(tmp_path)
    return result


"""
if __name__ == "__main__":
    download_vision_model()
"""
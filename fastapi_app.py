import os
import io
import torch
import torchaudio
import random
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from chatterbox.tts_turbo import ChatterboxTurboTTS
import uvicorn
from typing import Optional

app = FastAPI(
    title="Chatterbox TTS API",
    description="API for generating speech using Chatterbox Turbo TTS",
    version="1.0.0"
)

# Global model variable
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = None

def set_seed(seed: int):
    if seed == 0:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

@app.on_event("startup")
def load_model():
    global model
    print(f"--- Loading Chatterbox-Turbo on {DEVICE} ---")
    try:
        model = ChatterboxTurboTTS.from_pretrained(DEVICE)
        print("--- Model loaded successfully ---")
    except Exception as e:
        print(f"--- Error loading model: {e} ---")

# Determine the base directory for cloned voices (relative to this file)
CLONED_VOICES_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloned_voices")

@app.get("/voices")
async def list_voices():
    """Lists available cloned voices organized by gender."""
    voices = {"Man": [], "Woman": []}
    for gender in ["Man", "Woman"]:
        gender_path = os.path.join(CLONED_VOICES_BASE, gender)
        if os.path.exists(gender_path):
            files = os.listdir(gender_path)
            voices[gender] = [f.replace(".wav", "") for f in files if f.endswith(".wav")]
    return voices

@app.post("/generate")
async def generate_tts(
    text: str = Form(..., description="The text to synthesize into speech."),
    audio_prompt: Optional[UploadFile] = File(None, description="Optional 10s reference audio file for voice cloning."),
    voice_name: Optional[str] = Form(None, description="Name of the cloned voice to use (from /voices)."),
    gender: Optional[str] = Form(None, description="Gender of the cloned voice ('Man' or 'Woman')."),
    temperature: float = Form(0.8, description="Sampling temperature. Higher means more creative/random."),
    seed: int = Form(0, description="Random seed for reproducibility. 0 for random."),
    top_p: float = Form(0.95, description="Top-p sampling parameter."),
    top_k: int = Form(1000, description="Top-k sampling parameter."),
    repetition_penalty: float = Form(1.2, description="Penalty for repeating tokens."),
    norm_loudness: bool = Form(True, description="Whether to normalize the output loudness.")
):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded yet. Please wait.")

    temp_path = None
    try:
        set_seed(seed)
        
        audio_prompt_path = None
        
        # 1. Check if an audio file was uploaded (highest priority)
        if audio_prompt:
            temp_path = f"temp_{random.randint(1000, 9999)}_{audio_prompt.filename}"
            with open(temp_path, "wb") as f:
                content = await audio_prompt.read()
                f.write(content)
            audio_prompt_path = temp_path
            
        # 2. If no file uploaded, check for voice_name and gender
        elif voice_name and gender:
            # Handle case sensitivity for gender
            gender_dir = gender.capitalize() # Converts 'man' to 'Man', 'woman' to 'Woman'
            voice_file = f"{voice_name}.wav"
            potential_path = os.path.join(CLONED_VOICES_BASE, gender_dir, voice_file)
            
            if os.path.exists(potential_path):
                audio_prompt_path = potential_path
            else:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Voice '{voice_name}' not found for gender '{gender_dir}' in {CLONED_VOICES_BASE}"
                )

        # Generate audio using the model
        wav = model.generate(
            text,
            audio_prompt_path=audio_prompt_path,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            norm_loudness=norm_loudness,
        )

        # Cleanup temp file immediately after generation
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        # Convert the tensor to a WAV file in memory
        buffer = io.BytesIO()
        # Ensure we are saving a CPU tensor
        wav_cpu = wav.cpu()
        torchaudio.save(buffer, wav_cpu, model.sr, format="wav")
        buffer.seek(0)

        return StreamingResponse(
            buffer, 
            media_type="audio/wav", 
            headers={"Content-Disposition": "attachment; filename=output.wav"}
        )

    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Start server
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

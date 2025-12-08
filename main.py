from fastapi import FastAPI, UploadFile
from faster_whisper import WhisperModel
import requests
from joblib import load
import torch
from tripo3d import TripoClient, TaskStatus
import uvicorn
from pydantic import BaseModel
from google import genai
from google.genai import types
import uuid
import os
import base64
import aiohttp
import aiofiles
from fastapi import HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer
import asyncio
import io

app = FastAPI(title="Audio → Gemini → 3D com TripoSR")
TMP = "tmp"
os.makedirs(TMP, exist_ok=True)

TRIPO_API_KEY = os.getenv("TRIPO_API_KEY", "tsk_O8ofEfpK1ELfz3CvZ3uqp0NswE_DDanFjLm8hN1wPjI")

device = "cuda" if torch.cuda.is_available() else "cpu"

# stt model
stt_model = WhisperModel("base", device=device, compute_type="float32")

tokenizer = AutoTokenizer.from_pretrained("exported_model")
model = AutoModelForCausalLM.from_pretrained("exported_model")

# Unity request/response
class Room(BaseModel):
    width: int
    height: int

class Audio3DRequest(BaseModel):
    audio_base64: str
    objects: list[str]
    room: Room

class Position(BaseModel):
    name: str
    x: int
    z: int
    rot: int

class Layout3DResponse(BaseModel):
    output: list[Position]

# LLM Model resquest/response
class LLMRequest(BaseModel):
    prompt: str
    objects: list[str]
    room: Room

class LLMResponse(BaseModel):
    output: list[Position]

# --- Funções Auxiliares ---

# --- Configurações de Polling ---
MAX_RETRIES = 60  # Tenta por 60 vezes (1 minuto)
POLLING_DELAY = 1 # Espera 1 segundo entre as tentativas

async def generate_3d_with_tripo(prompt: str, output_path: str):
    async with TripoClient(api_key=TRIPO_API_KEY) as client:
        task_id = await client.text_to_model(
            prompt=prompt
        )
        print(f"Task ID: {task_id}")

        task = await client.wait_for_task(task_id, verbose=True)
        if task.status == TaskStatus.SUCCESS:
            files = await client.download_task_models(task, output_path)
            for model_type, path in files.items():
                print(f"Downloaded {model_type}: {path}")

# -----------------------------------
# ENDPOINT PRINCIPAL: Audio → 3D
# -----------------------------------
@app.post("/generate-3d", response_model=Layout3DResponse)
async def generate_3d(req: Audio3DRequest):
    # 1) SALVAR ÁUDIO TEMPORÁRIO
    audio_id = str(uuid.uuid4())
    audio_path = f"{TMP}/{audio_id}.wav"
    image_path = f"{TMP}/{audio_id}.png"
    glb_path = f"{TMP}/{audio_id}.glb"

    audio_bytes = base64.b64decode(req.audio_base64)
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)

    # 2) STT → TEXTO
    segments, _ = stt_model.transcribe(audio_path)
    prompt = " ".join([s.text for s in segments]).strip()

    if not prompt:
            raise HTTPException(status_code=400, detail="Não foi possível transcrever áudio.")

    # O prompt é refinado para garantir melhor resultado 3D
    enhanced_prompt = f"Full body 3D render of {prompt}, clean background, high quality, 3d asset style"

    inputs = tokenizer(enhanced_prompt, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=300)
    result = tokenizer.decode(output[0], skip_special_tokens=True)

    # 5) CONVERTER ARQUIVOS PARA BASE64
    async with aiofiles.open(image_path, "rb") as f:
        image_b64 = base64.b64encode(await f.read()).decode()

    async with aiofiles.open(glb_path, "rb") as f:
        glb_b64 = base64.b64encode(await f.read()).decode()


# -----------------------------
# Iniciar servidor
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)

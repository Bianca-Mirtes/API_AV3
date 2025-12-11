import math
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn

# =========================
# CONFIGURAÇÕES
# =========================

MODEL_PATH = "layout_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_OBJECTS = 7

# =========================
# DEFINIÇÃO DOS OBJETOS
# (TEM QUE SER IGUAL AO TREINO)
# =========================

OBJ_TYPES = sorted([
    "toilet", "sink", "shower",
    "fridge", "oven", "sink_kitchen", "table",
    "bed", "wardrobe", "nightstand",
    "sofa", "tv", "coffee_table"
])

OBJ_TO_IDX = {o: i for i, o in enumerate(OBJ_TYPES)}
NUM_OBJ_TYPES = len(OBJ_TYPES)

INPUT_SIZE = 2 + MAX_OBJECTS * (NUM_OBJ_TYPES + 2)
OUTPUT_SIZE = MAX_OBJECTS * 4

# =========================
# MODELO
# =========================

class LayoutMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, OUTPUT_SIZE)
        )

    def forward(self, x):
        return self.net(x)

model = LayoutMLP().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# =========================
# FASTAPI
# =========================

app = FastAPI(title="Layout 3D Predictor")

# =========================
# MODELOS DE ENTRADA
# =========================

class ObjectInput(BaseModel):
    type: str
    size: List[float]  # [width, depth]

class LayoutRequest(BaseModel):
    room: List[float]  # [width, depth]
    objects: List[ObjectInput]

# =========================
# PREPROCESSAMENTO
# =========================

def preprocess(req: LayoutRequest):
    room_w, room_d = req.room
    x = [room_w, room_d]

    objs = req.objects[:MAX_OBJECTS]

    for obj in objs:
        onehot = [0.0] * NUM_OBJ_TYPES
        if obj.type in OBJ_TO_IDX:
            onehot[OBJ_TO_IDX[obj.type]] = 1.0

        x.extend(onehot)
        x.extend([obj.size[0], obj.size[1]])

    while len(objs) < MAX_OBJECTS:
        x.extend([0.0] * NUM_OBJ_TYPES)
        x.extend([0.0, 0.0])
        objs.append(None)

    return torch.tensor(x, dtype=torch.float32).unsqueeze(0)

# =========================
# PÓS-PROCESSAMENTO
# =========================

def postprocess(pred, room_w, room_d, n_objects):
    pred = pred.squeeze(0).cpu().numpy()
    results = []

    for i in range(n_objects):
        base = i * 4
        x_norm = pred[base]
        z_norm = pred[base + 1]
        sin_t = pred[base + 2]
        cos_t = pred[base + 3]

        x = float(x_norm * room_w)
        z = float(z_norm * room_d)
        angle = math.degrees(math.atan2(sin_t, cos_t))

        results.append({
            "x": x,
            "z": z,
            "rotation": angle
        })

    return results

# =========================
# ENDPOINT
# =========================

@app.post("/predict_layout")
def predict_layout(req: LayoutRequest):
    room_w, room_d = req.room
    x = preprocess(req).to(DEVICE)

    with torch.no_grad():
        pred = model(x)

    return postprocess(
        pred,
        room_w,
        room_d,
        n_objects=len(req.objects)
    )

if __name__ == "__main__":
    print("="*60)
    print("🚀 API Layout 3D Generator")
    print("="*60)
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8081,
        log_level="info"
    )

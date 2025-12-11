import json 
import math 
import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader 
from pathlib import Path 

# ========================= # CONFIGURAÇÕES GERAIS # ========================= 
DATASET_PATH = "dataset.jsonl" 
MODEL_OUT = "layout_model.pt" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
MAX_OBJECTS = 7
BATCH_SIZE = 32 
EPOCHS = 80 
LEARNING_RATE = 1e-3 

# ========================= # DEFINIÇÃO DOS OBJETOS # ========================= 
OBJ_TYPES = sorted([ "toilet", "sink", "shower", "fridge", "oven", "sink_kitchen", "table", "bed", "wardrobe", "nightstand", "sofa", "tv", "coffee_table" ]) 
OBJ_TO_IDX = {o: i for i, o in enumerate(OBJ_TYPES)} 
NUM_OBJ_TYPES = len(OBJ_TYPES) 

# === INPUT ===
# room_w, room_d → 2 valores
# por objeto → onehot(13) + w + d + h → 16 valores
INPUT_PER_OBJECT = NUM_OBJ_TYPES + 3
INPUT_SIZE = 3 + MAX_OBJECTS * INPUT_PER_OBJECT

# === OUTPUT ===
# para cada objeto → x, y, z, sin(rot), cos(rot) = 5 valores
OUTPUT_PER_OBJECT = 5
OUTPUT_SIZE = MAX_OBJECTS * OUTPUT_PER_OBJECT

# ========================= # ENCODING DA CENA # ========================= 
def encode_scene(scene):
    room_w, room_h, room_d = scene["room"]

    # Agora 3 valores da sala
    x = [room_w, room_h, room_d]

    objects = scene["objects"][:MAX_OBJECTS]

    # INPUT — exatamente 16 por objeto
    for obj in objects:
        onehot = [0.0] * NUM_OBJ_TYPES
        onehot[OBJ_TO_IDX[obj["type"]]] = 1.0

        w = obj.get("w", 0.0)
        d = obj.get("d", 0.0)
        h = obj.get("h", 0.0)

        x.extend(onehot)
        x.extend([w, d, h])

    # PADDING — garante 16 * MAX_OBJECTS
    for _ in range(MAX_OBJECTS - len(objects)):
        x.extend([0.0] * NUM_OBJ_TYPES)
        x.extend([0.0, 0.0, 0.0])

    # TARGET (sempre 5 * MAX_OBJECTS)
    y = []
    for obj in objects:
        x_norm = obj["x"] / room_w
        y_norm = obj["y"] / room_h
        z_norm = obj["z"] / room_d
        r = math.radians(obj["rot"])
        y.extend([x_norm, y_norm, z_norm, math.sin(r), math.cos(r)])

    # Padding para targets faltantes
    while len(y) < OUTPUT_SIZE:
        y.extend([0.0, 0.0, 0.0, 0.0, 0.0])

    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
        
# ========================= # DATASET PYTORCH # ========================= 
class LayoutDataset(Dataset): 
    def __init__(self, json_path): 
        with open(json_path, "r") as f: 
            self.data = json.load(f) 

    def __len__(self): 
        return len(self.data) 

    def __getitem__(self, idx): 
        return encode_scene(self.data[idx]) 

# ========================= # MODELO MLP # ========================= 
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
                
def collision_loss(pred, sizes):
    loss = 0.0

    for i in range(MAX_OBJECTS):
        for j in range(i + 1, MAX_OBJECTS):
            # x/z normalizados → desnormalizar
            dx = torch.abs(pred[i, 0] - pred[j, 0])
            dz = torch.abs(pred[i, 2] - pred[j, 2])

            min_x = (sizes[i][0] + sizes[j][0]) / 2
            min_z = (sizes[i][1] + sizes[j][1]) / 2

            overlap_x = torch.relu(min_x - dx)
            overlap_z = torch.relu(min_z - dz)

            loss += overlap_x * overlap_z

    return loss

# =========================
# Loss de espalhamento
# =========================
def spread_loss(pred):
    loss = 0.0
    for i in range(MAX_OBJECTS):
        for j in range(i + 1, MAX_OBJECTS):
            dx = pred[i, 0] - pred[j, 0]
            dz = pred[i, 2] - pred[j, 2]
            dist = torch.sqrt(dx * dx + dz * dz + 1e-6)
            loss += torch.relu(0.5 - dist)
    return loss

# ========================= 
# TREINAMENTO 
# =========================
def train():
    dataset = LayoutDataset(DATASET_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = LayoutMLP().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)

            pos_loss = torch.mean((pred - y) ** 2)

            # tamanhos dos objetos
            sizes = []
            idx = 2
            for _ in range(MAX_OBJECTS):
                idx += NUM_OBJ_TYPES
                w = x[0, idx]
                d = x[0, idx + 1]
                sizes.append((w, d))
                idx += 3

            pred_first = pred[0].reshape(MAX_OBJECTS, OUTPUT_PER_OBJECT)

            col_loss = collision_loss(pred_first, sizes)
            spr_loss = spread_loss(pred_first)

            loss = pos_loss + 5.0 * col_loss + 0.5 * spr_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), MODEL_OUT)

# =========================
# EXECUÇÃO 
# =========================
if __name__ == "__main__": 
    train()
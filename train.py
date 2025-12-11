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
INPUT_SIZE = 2 + MAX_OBJECTS * (NUM_OBJ_TYPES + 2) 
OUTPUT_SIZE = MAX_OBJECTS * 4 
# ========================= # ENCODING DA CENA # ========================= 
def encode_scene(scene): 
    """
    Entrada:
      - room width, depth
      - para cada objeto:
          one-hot(type) + width + depth

    Saída:
      - para cada objeto:
          x_norm, z_norm, sin(theta), cos(theta)
    """
    room_w, room_d = scene["room"] 
    x = [room_w, room_d] 

    objects = scene["objects"][:MAX_OBJECTS]

    # input
    for obj in objects: 
        onehot = [0.0] * NUM_OBJ_TYPES 
        onehot[OBJ_TO_IDX[obj["type"]]] = 1.0

        x.extend(onehot) 
        x.extend([obj["w"], obj["d"]]) 

    # padding    
    while len(objects) < MAX_OBJECTS: 
        x.extend([0.0] * NUM_OBJ_TYPES) 
        x.extend([0.0, 0.0]) 
        objects.append(None) 

    # TARGET 
    y = [] 
    for obj in scene["objects"][:MAX_OBJECTS]: 
        y.extend([ 
            obj["x"] / room_w, 
            obj["z"] / room_d, 
            math.sin(math.radians(obj["rot"])), 
            math.cos(math.radians(obj["rot"])) 
            ]) 

    while len(y) < OUTPUT_SIZE:
        y.extend([0.0, 0.0, 0.0, 0.0]) 
    
    return ( 
        torch.tensor(x, dtype=torch.float32), 
        torch.tensor(y, dtype=torch.float32) 
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
            nn.ReLU(), nn.Linear(512, 512), 
            nn.ReLU(), nn.Linear(512, 256), 
            nn.ReLU(), nn.Linear(256, OUTPUT_SIZE) 
        ) 
    def forward(self, x): 
        return self.net(x)
                
def collision_loss(pred, sizes, n_objs):
    loss = 0.0
    for i in range(n_objs):
        for j in range(i + 1, n_objs):
            bi = i * 4
            bj = j * 4

            dx = torch.abs(pred[bi] - pred[bj])
            dz = torch.abs(pred[bi + 1] - pred[bj + 1])

            min_x = (sizes[i][0] + sizes[j][0]) / 2
            min_z = (sizes[i][1] + sizes[j][1]) / 2

            overlap_x = torch.relu(min_x - dx)
            overlap_z = torch.relu(min_z - dz)

            loss += overlap_x * overlap_z
    return loss

def spread_loss(pred, n): 
    loss = 0 
    for i in range(n): 
        for j in range(i+1, n): 
            dx = pred[i*3] - pred[j*3] 
            dz = pred[i*3+1] - pred[j*3+1] 
            dist = torch.sqrt(dx*dx + dz*dz + 1e-6) 
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
            n_objs = MAX_OBJECTS
            idx = 2
            for _ in range(MAX_OBJECTS):
                idx += NUM_OBJ_TYPES
                w = x[0, idx]
                d = x[0, idx + 1]
                sizes.append((w, d))
                idx += 2

            col_loss = collision_loss(pred[0], sizes, n_objs)
            spr_loss = spread_loss(pred[0], n_objs)

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
if __name__ == "__main__": train()
# === Colab-ready: Treino Phi-3 + LoRA (pré-tokenizado, dataset limpo) ===

# 0) Instalação (execute primeira vez)
!pip install -q -U "torch>=2.2.1" transformers==4.38.2 accelerate==0.27.2 bitsandbytes==0.43.1 peft==0.10.0 datasets

# 1) Imports
import os
import json
import random
import math
from pathlib import Path

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 2) Config (ajuste aqui se precisar)
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
NUM_EXAMPLES = 3000
DATASET_RAW = "dataset.jsonl"            # se já tiver, comente a geração e mantenha
DATASET_CLEAN = "dataset_cleaned.jsonl"
EXPORT_DIR = "exported_model"

MAX_LENGTH = 192        # reduzido para velocidade (ajuste entre 160..256)
MAX_STEPS = 300         # ajuste conforme tempo disponível
PER_DEVICE_BS = 2
GRAD_ACCUM = 8
LORA_R = 4

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# 3) Gerador de dataset (limpo, seguro)
# (Se você já tem dataset.jsonl, isso sobrescreve dataset.jsonl com versão corrigida — se não quiser, pule esta seção.)
OBJECT_CATALOG = {
    "mesa": "furniture",
    "cadeira": "furniture",
    "pc": "electronics",
    "sofa": "furniture",
    "televisao": "electronics",
    "cama": "furniture",
    "estante": "furniture",
    "armario": "furniture",
    "tapete": "decoration",
    "luminaria": "lighting",
    "mesa_de_centro": "furniture",
    "geladeira": "appliances",
    "fogao": "appliances",
    "pia": "appliances",
    "espelho": "decoration",
    "chuveiro": "plumbing",
    "privada": "plumbing"
}

ROOM_ALLOWED_OBJECTS = {
    "sala gamer minimalista": {"mesa","cadeira","pc","sofa","televisao","luminaria","tapete","estante"},
    "sala de estar moderna": {"sofa","televisao","mesa_de_centro","tapete","luminaria","estante"},
    "quarto simples": {"cama","armario","luminaria","espelho","tapete"},
    "escritorio pequeno": {"mesa","cadeira","pc","luminaria","estante","armario"},
    "home office organizado": {"mesa","cadeira","pc","luminaria","estante"},
    "sala compacta": {"sofa","televisao","mesa_de_centro","tapete","estante"},
    "banheiro compacto": {"pia","chuveiro","privada","espelho"},
    "cozinha moderna": {"fogao","geladeira","pia","armario","mesa"},
    "cozinha americana": {"fogao","geladeira","pia","mesa","cadeira"},
    "quarto moderno": {"cama","armario","espelho","luminaria"},
    "sala de TV": {"sofa","televisao","mesa_de_centro","tapete"}
}

INSTRUCTION_TEMPLATES = [
    "quero montar um {room_style} para trabalhar horas seguidas",
    "organize uma {room_style} com foco em conforto",
    "quero montar um {room_style}",
    "organize um {room_style} para mim",
    "como ficaria a disposição de um {room_style}?",
    "crie um layout funcional para um {room_style}",
    "me ajude a organizar um {room_style}"
]
ROOM_STYLES = list(ROOM_ALLOWED_OBJECTS.keys())

def clamp(v, a, b): return max(a, min(b, v))
def normalize_rot(rot): return int(((rot + 180) % 360) - 180)
def distance(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])

def generate_room_dimensions():
    return {"width": round(random.uniform(3,8),1), "depth": round(random.uniform(3,8),1)}

def pick_objects_for(style):
    allowed = ROOM_ALLOWED_OBJECTS[style]
    n = random.randint(3, min(len(allowed),7))
    return random.sample(list(allowed), n)

def generate_instruction(style):
    template = random.choice(INSTRUCTION_TEMPLATES)
    return template.format(room_style=style)

def nearest_wall_rotation(x, z, room):
    hw, hd = room["width"]/2, room["depth"]/2
    distances = [(abs(x+hw),90),(abs(x-hw),-90),(abs(z+hd),0),(abs(z-hd),180)]
    _, rot = min(distances, key=lambda a:a[0])
    return normalize_rot(rot)

def place_near(x_base, z_base, room, distance=0.9):
    angle = random.uniform(0, 2*math.pi)
    x = x_base + math.cos(angle) * distance
    z = z_base + math.sin(angle) * distance
    hw, hd = room["width"]/2, room["depth"]/2
    x = round(clamp(x, -hw+0.05, hw-0.05), 2)
    z = round(clamp(z, -hd+0.05, hd-0.05), 2)
    return x,z

def is_far_enough(x,z, used, min_dist=0.9):
    return all(distance((x,z), p) > min_dist for p in used)

def try_place(room, used):
    hw, hd = room["width"]/2, room["depth"]/2
    for _ in range(40):
        x = random.uniform(-hw, hw)
        z = random.uniform(-hd, hd)
        if is_far_enough(x,z, used):
            rot = nearest_wall_rotation(x,z,room)
            return round(x,2), round(z,2), rot
    # fallback clamped
    x = round(clamp(random.uniform(-hw, hw), -hw+0.05, hw-0.05),2)
    z = round(clamp(random.uniform(-hd, hd), -hd+0.05, hd-0.05),2)
    rot = nearest_wall_rotation(x,z,room)
    return x,z,rot

def generate_layout_positions(objects, room):
    positions=[]
    used=[]
    mesa=None
    tv=None
    for obj in objects:
        if obj=="pc" and mesa:
            positions.append({"name":"pc","x":mesa["x"],"y":0,"z":mesa["z"],"rot":mesa["rot"]})
            continue
        if obj=="cadeira" and mesa:
            x,z = place_near(mesa["x"], mesa["z"], room, distance=0.9)
            rot = nearest_wall_rotation(x,z,room)
            used.append((x,z))
            positions.append({"name":"cadeira","x":x,"y":0,"z":z,"rot":rot})
            continue
        if obj=="sofa" and tv:
            x,z = place_near(tv["x"], tv["z"], room, distance=2.0)
            rot = nearest_wall_rotation(x,z,room)
            used.append((x,z))
            positions.append({"name":"sofa","x":x,"y":0,"z":z,"rot":rot})
            continue
        x,z,rot = try_place(room, used)
        used.append((x,z))
        pos={"name":obj,"x":x,"y":0,"z":z,"rot":rot}
        positions.append(pos)
        if obj=="mesa": mesa=pos
        if obj=="televisao": tv=pos
    return positions

def generate_example():
    style = random.choice(ROOM_STYLES)
    objects = pick_objects_for(style)
    room = generate_room_dimensions()
    instr = generate_instruction(style)
    output = generate_layout_positions(objects, room)
    return {"instruction": instr, "objects": objects, "room": room, "output": output}

# generate cleaned dataset
print("Gerando dataset limpo...")
with open(DATASET_CLEAN, "w", encoding="utf-8") as fout:
    for i in range(NUM_EXAMPLES):
        ex = generate_example()
        fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
        if (i+1) % 200 == 0:
            print(f"  -> {i+1}/{NUM_EXAMPLES}")
print("Dataset limpo salvo em", DATASET_CLEAN)

# 4) Carregar dataset limpo e pré-tokenizar (rápido, em batch)
print("Carregando dataset ...")
ds = load_dataset("json", data_files=DATASET_CLEAN)["train"]

# montar prompt (entrada) + resposta (output JSON)
def build_io(example):
    prompt = f"Você é um modelo que organiza layouts 3D para Unity.\nInstrução: {example['instruction']}\n\nResposta:"
    out = json.dumps(example["output"], ensure_ascii=False)
    text = prompt + out
    return {"text": text}

ds = ds.map(build_io, batched=False)

print("Carregando tokenizer e pré-tokenizando (padding max_length=%d)..." % MAX_LENGTH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
# pad right for efficiency; Phi-3 tokenizers are often left but right is fine for causal when prepadding fixed
tokenizer.padding_side = "right"

def tokenize_batch(batch):
    enc = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
    enc["labels"] = enc["input_ids"].copy()
    return enc

# tokenizar em vários processos se possível (ajuste num_proc conforme seu ambiente)
ds_tokenized = ds.map(tokenize_batch, batched=True, batch_size=256, num_proc=2)
ds_tokenized = ds_tokenized.remove_columns(["text"])
ds_tokenized.set_format(type="torch")

print("Tokenização completa. Exemplos tokenizados:", len(ds_tokenized))

# 5) Carregar modelo quantizado + preparar LoRA
print("Carregando modelo quantizado 4-bit (pode demorar)...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb, device_map="auto")
model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","down_proj","up_proj"]
)

model = get_peft_model(model, lora_cfg)

# 6) TrainingArguments otimizados
training_args = TrainingArguments(
    output_dir="./layout-model",
    per_device_train_batch_size=PER_DEVICE_BS,
    gradient_accumulation_steps=GRAD_ACCUM,
    max_steps=MAX_STEPS,
    learning_rate=2e-4,
    logging_steps=100,
    save_strategy="no",
    fp16=True,
    optim="adamw_torch",
    report_to="none"
)

# 7) Trainer (dataset já padronizado, não precisa de data_collator)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_tokenized
)

# 8) Treinar
print("Iniciando treino...")
trainer.train()

# 9) Salvar modelo + tokenizer
print("Salvando modelo em", EXPORT_DIR)
model.save_pretrained(EXPORT_DIR)
tokenizer.save_pretrained(EXPORT_DIR)
print("Pronto. Modelo salvo em", EXPORT_DIR)
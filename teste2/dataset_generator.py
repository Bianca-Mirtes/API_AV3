import json
import random
import math

# ---- CONFIG ---- #
NUM_EXAMPLES = 3000

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

INSTRUCTION_TEMPLATES = [
    "quero montar um {room_style} para trabalhar horas seguidas",
    "organize uma {room_style} com foco em conforto",
    "quero montar um {room_style}",
    "organize um {room_style} para mim",
    "como ficaria a disposição de um {room_style}?",
    "crie um layout funcional para um {room_style}",
    "me ajude a organizar um {room_style}"
]

ROOM_STYLES = [
    "sala gamer minimalista",
    "sala de estar moderna",
    "quarto simples",
    "escritório pequeno",
    "home office organizado",
    "sala compacta",
    "banheiro compacto",
    "cozinha moderna",
    "cozinha americana",
    "quarto moderno",
    "sala de TV"
]

# ---------------------------------------------------
#               AUXILIARY FUNCTIONS
# ---------------------------------------------------

def generate_room_dimensions():
    width = round(random.uniform(2, 8), 1)
    depth = round(random.uniform(2, 8), 1)
    return {"width": width, "depth": depth}

def pick_random_objects():
    keys = list(OBJECT_CATALOG.keys())
    n = random.randint(3, 7)
    return random.sample(keys, n)

def generate_instruction():
    template = random.choice(INSTRUCTION_TEMPLATES)
    style = random.choice(ROOM_STYLES)
    return template.format(room_style=style)

def nearest_wall_rotation(x, z, room):
    """Define a rotação automática baseada na parede mais próxima."""
    half_w, half_d = room["width"] / 2, room["depth"] / 2

    dist_left = abs(x + half_w)
    dist_right = abs(x - half_w)
    dist_front = abs(z + half_d)
    dist_back = abs(z - half_d)

    min_dist = min(dist_left, dist_right, dist_front, dist_back)

    if min_dist == dist_left:
        return 90   # olhando para direita
    elif min_dist == dist_right:
        return -90  # olhando para esquerda
    elif min_dist == dist_front:
        return 0    # olhando para trás
    else:
        return 180  # olhando para frente

def place_object_near(x_base, z_base, distance=0.8):
    """Posiciona algo atrás ou na frente de outro objeto."""
    angle = random.uniform(0, math.pi * 2)
    x = round(x_base + math.cos(angle) * distance, 2)
    z = round(z_base + math.sin(angle) * distance, 2)
    return x, z

def avoid_collisions(x, z, used_positions):
    return all(abs(x - up[0]) > 1.0 and abs(z - up[1]) > 1.0 for up in used_positions)

# ---------------------------------------------------
#               MAIN LAYOUT GENERATOR
# ---------------------------------------------------
MAX_ATTEMPTS = 40   # evita loops infinitos

def generate_layout_positions(objects, room):
    positions = []
    used_positions = []

    mesa_pos = None
    tv_pos = None
    cama_pos = None

    def try_place_random():
        """Tenta colocar um objeto em posição aleatória evitando colisões."""
        for _ in range(MAX_ATTEMPTS):
            x = round(random.uniform(-room["width"]/2, room["width"]/2), 2)
            z = round(random.uniform(-room["depth"]/2, room["depth"]/2), 2)

            if avoid_collisions(x, z, used_positions):
                rot = nearest_wall_rotation(x, z, room)
                return x, z, rot

        # Fallback: coloca mesmo que colida
        x = round(random.uniform(-room["width"]/2, room["width"]/2), 2)
        z = round(random.uniform(-room["depth"]/2, room["depth"]/2), 2)
        rot = nearest_wall_rotation(x, z, room)
        return x, z, rot


    for obj in objects:

        # =========================================
        # PC EM CIMA DA MESA
        # =========================================
        if obj == "pc" and "mesa" in objects:
            if mesa_pos:
                positions.append({
                    "name": "pc",
                    "x": mesa_pos["x"],
                    "z": mesa_pos["z"],
                    "rot": mesa_pos["rot"]
                })
                continue

        # =========================================
        # CADEIRA ATRÁS DA MESA
        # =========================================
        if obj == "cadeira" and "mesa" in objects:
            if mesa_pos:
                x, z = place_object_near(mesa_pos["x"], mesa_pos["z"], distance=0.9)
                rot = nearest_wall_rotation(x, z, room)
                used_positions.append((x, z))
                positions.append({"name": obj, "x": x, "z": z, "rot": rot})
                continue

        # =========================================
        # SOFÁ OLHANDO PARA A TV
        # =========================================
        if obj == "sofa" and "televisao" in objects and tv_pos:
            x, z = place_object_near(tv_pos["x"], tv_pos["z"], distance=2.0)
            rot = nearest_wall_rotation(x, z, room)
            used_positions.append((x, z))
            positions.append({"name": obj, "x": x, "z": z, "rot": rot})
            continue

        # =========================================
        # CAMA (encostada na parede)
        # =========================================
        if obj == "cama":
            x, z, rot = try_place_random()
            cama_pos = {"name": obj, "x": x, "z": z, "rot": rot}
            used_positions.append((x, z))
            positions.append(cama_pos)
            continue

        # =========================================
        # OBJETOS NORMAIS
        # =========================================
        x, z, rot = try_place_random()
        used_positions.append((x, z))
        this_pos = {"name": obj, "x": x, "z": z, "rot": rot}
        positions.append(this_pos)

        # salvar para regras especiais
        if obj == "mesa":
            mesa_pos = this_pos
        if obj == "televisao":
            tv_pos = this_pos

    return positions

# ---------------------------------------------------
#               EXAMPLE GENERATOR
# ---------------------------------------------------

def generate_example():
    objects = pick_random_objects()
    room = generate_room_dimensions()
    instruction = generate_instruction()
    output = generate_layout_positions(objects, room)

    return {
        "instruction": instruction,
        "objects": objects,
        "room": room,
        "output": output
    }

# ---------------------------------------------------
#               DATASET GENERATION
# ---------------------------------------------------

if __name__ == "__main__":
    with open("dataset.jsonl", "w", encoding="utf-8") as f:
        for i in range(NUM_EXAMPLES):
            if i % 100 == 0:
                print("Gerando exemplo:", i)
            example = generate_example()
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"Dataset with {NUM_EXAMPLES} examples generated in 'dataset.jsonl'")
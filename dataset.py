import random
import json
import math
from pathlib import Path

ROOM_TYPES = {
    "bathroom": ["toilet", "sink", "shower"],
    "kitchen": ["fridge", "oven", "sink_kitchen", "table"],
    "bedroom": ["bed", "wardrobe", "nightstand"],
    "livingroom": ["sofa", "tv", "coffee_table"]
}

# Objetos sobre quais outros podem ficar
SUPPORT_MAP = {
    "tv": ["coffee_table", "table", "wardrobe", "nightstand"]
}

OBJECT_SIZES = {
    "toilet": (1, 1, 1),
    "sink": (1, 1, 1),
    "shower": (1.0, 1.0, 2.0),

    "fridge": (1, 1, 1.8),
    "oven": (1, 1, 1.0),
    "sink_kitchen": (1, 1, 1),
    "table": (1.2, 1.2, 1),

    "bed": (2.0, 1.6, 1),
    "wardrobe": (1.8, 1, 2.0),
    "nightstand": (1, 1, 1),

    "sofa": (2.0, 1, 1),
    "tv": (1.2, 1, 1),
    "coffee_table": (1, 1, 1)
}

ALL_OBJECTS = list(OBJECT_SIZES.keys())

def place_on_top(child_name, objects):
    """
    Tenta colocar o child em cima de um objeto adequado.
    Retorna (x, y, z) ou None se não possível.
    """
    if child_name not in SUPPORT_MAP:
        return None

    valid_supports = SUPPORT_MAP[child_name]

    possible_supports = [o for o in objects if o["type"] in valid_supports]
    if not possible_supports:
        return None

    # Escolhe um suporte aleatório
    support = random.choice(possible_supports)

    # centraliza o objeto filho no suporte
    x = support["x"]
    z = support["z"]

    # y = topo do suporte
    y = support["h"]

    return x, y, z

def aabb_collision(a, b):
    return not (
        a["x"] + a["w"]/2 < b["x"] - b["w"]/2 or
        a["x"] - a["w"]/2 > b["x"] + b["w"]/2 or
        a["z"] + a["d"]/2 < b["z"] - b["d"]/2 or
        a["z"] - a["d"]/2 > b["z"] + b["d"]/2
    )

def collides(obj, others):
    for o in others:
        if aabb_collision(obj, o):
            return True
    return False

def generate_scene(max_objects=7):
    room_w = random.uniform(4.0, 8.0)
    room_d = random.uniform(4.0, 8.0)
    room_h = random.uniform(room_w*room_d, room_w*room_d + 4.0)

    room_type = random.choice(list(ROOM_TYPES.keys()))
    candidates = ROOM_TYPES[room_type]

    n_objects = random.randint(2, min(max_objects, len(candidates)))
    chosen = random.sample(candidates, n_objects)

    objects = []

    # centro semântico do cômodo
    cluster_x = random.uniform(1.5, room_w - 1.5)
    cluster_z = random.uniform(1.5, room_d - 1.5)

    for obj_name in chosen:
        w, d, h = OBJECT_SIZES[obj_name]

        # 1 — tenta colocar em cima de um suporte existente
        support_pos = place_on_top(obj_name, objects)
        if support_pos:
            x, y, z = support_pos
            rot = random.choice([0, 90, 180, 270])

            obj = {
                "type": obj_name,
                "room": room_type,
                "x": x,
                "y": y,
                "z": z,
                "w": w,
                "d": d,
                "h": h,
                "rot": rot,
                "supported": True
            }
            objects.append(obj)
            continue

        # 2 — caso sem suporte → coloca no chão como antes
        for _ in range(50):
            x = random.gauss(cluster_x, 0.6)
            z = random.gauss(cluster_z, 0.6)

            x = max(w/2, min(room_w - w/2, x))
            z = max(d/2, min(room_d - d/2, z))

            rot = random.choice([0, 90, 180, 270])

            candidate = {
                "type": obj_name,
                "room": room_type,
                "x": x,
                "y": 0.0,
                "z": z,
                "w": w,
                "d": d,
                "h": h,
                "rot": rot,
                "supported": False
            }
            print(candidate)

            if not collides(candidate, objects):
                objects.append(candidate)
                break

    return {
        "room": [room_w, room_h, room_d],
        "room_type": room_type,
        "objects": objects
    }

def generate_dataset(n_samples=8000, out_file="dataset.jsonl"):
    data = []
    for i in range(n_samples):
        scene = generate_scene()
        data.append(scene)

    with open(out_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Dataset salvo com {len(data)} cenas.")

if __name__ == "__main__":
    generate_dataset()

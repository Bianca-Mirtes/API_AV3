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

OBJECT_SIZES = {
    "toilet": (0.6, 0.6),
    "sink": (0.5, 0.4),
    "shower": (1.0, 1.0),

    "fridge": (0.8, 0.8),
    "oven": (0.6, 0.6),
    "sink_kitchen": (0.6, 0.4),
    "table": (1.2, 1.2),

    "bed": (2.0, 1.6),
    "wardrobe": (1.8, 0.6),
    "nightstand": (0.5, 0.5),

    "sofa": (2.0, 0.8),
    "tv": (1.2, 0.3),
    "coffee_table": (0.8, 0.6)
}

ALL_OBJECTS = list(OBJECT_SIZES.keys())

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

    room_type = random.choice(list(ROOM_TYPES.keys()))
    candidates = ROOM_TYPES[room_type]

    n_objects = random.randint(2, min(max_objects, len(candidates)))
    chosen = random.sample(candidates, n_objects)

    objects = []

    # centro semântico do cômodo
    cluster_x = random.uniform(1.5, room_w - 1.5)
    cluster_z = random.uniform(1.5, room_d - 1.5)

    for obj_name in chosen:
        w, d = OBJECT_SIZES[obj_name]

        for _ in range(50):  # tentativas
            x = random.gauss(cluster_x, 0.6)
            z = random.gauss(cluster_z, 0.6)

            x = max(w/2, min(room_w - w/2, x))
            z = max(d/2, min(room_d - d/2, z))

            rot = random.choice([0, 90, 180, 270])

            candidate = {
                "type": obj_name,
                "room": room_type,
                "x": x,
                "z": z,
                "w": w,
                "d": d,
                "rot": rot
            }

            if not collides(candidate, objects):
                objects.append(candidate)
                break

    return {
        "room": [room_w, room_d],
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

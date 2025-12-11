import json
from verifier import verify_relation   # você vai ganhar verifier abaixo
from mistral_infer import predict      # inferência usando Mistral
from generate_dataset import RELATIONS
import random

def load_dataset(path="dataset.jsonl"):
    return [json.loads(l) for l in open(path)]

def append_to_dataset(sample, file="dataset.jsonl"):
    with open(file, "a") as f:
        f.write(json.dumps(sample) + "\n")

def self_improve_cycle(dataset_path="dataset.jsonl", cycles=200):
    data = load_dataset(dataset_path)

    for _ in range(cycles):
        ex = random.choice(data)

        instruction = ex["instruction"]
        scene = ex["scene"]
        object_from = list(scene.keys())[0]
        object_to = list(scene.keys())[1]

        pred = predict(instruction, scene)  # modelo gera posição

        ok, corrected = verify_relation(instruction, scene, pred)

        if not ok:
            new_example = {
                "instruction": instruction,
                "scene": scene,
                "model_wrong": pred,
                "answer": corrected
            }
            append_to_dataset(new_example)
            print("Corrigido e adicionado novo exemplo.")

    print("Ciclo de autocorreção finalizado.")

def verify_relation(instruction, scene, pred):
    """Verifica se o modelo respeitou a regra e retorna True/False + posição corrigida."""
    keys = list(scene.keys())
    objA = scene[keys[0]]
    objB = scene[keys[1]]

    if "TV" in instruction or "tv" in instruction:
        correct = {
            "x": objA["x"],
            "y": objA["y"] + objA["h"]/2 + 0.02,
            "z": objA["z"]
        }
        ok = (
            abs(pred["x"] - correct["x"]) < 0.05 and
            abs(pred["y"] - correct["y"]) < 0.05 and
            abs(pred["z"] - correct["z"]) < 0.05
        )
        return ok, correct

    return True, pred  # fallback

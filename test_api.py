import requests
import math
import time

# =========================
# CONFIGURAÇÕES
# =========================

API_URL = "http://localhost:8081/predict"

# Cena de teste controlada
TEST_SCENE = {
    "room": [6.0, 5.0],
    "objects": [
        {"type": "sofa", "size": [2.0, 0.8]},
        {"type": "tv", "size": [1.2, 0.3]},
        {"type": "coffee_table", "size": [0.8, 0.6]}
    ]
}

# =========================
# FUNÇÕES DE VALIDAÇÃO
# =========================

def validate_response(scene, response):
    room_w, room_d = scene["room"]
    objects = scene["objects"]

    assert isinstance(response, list), "Resposta não é uma lista"
    assert len(response) == len(objects), "Número de objetos não bate"

    for i, obj in enumerate(response):
        x = obj["x"]
        z = obj["z"]
        rot = obj["rotation"]

        # Verificações básicas
        assert not math.isnan(x), "x é NaN"
        assert not math.isnan(z), "z é NaN"
        assert not math.isnan(rot), "rotação é NaN"

        # Dentro da sala
        assert 0.0 <= x >= room_w, f"x fora da sala: {x}"
        assert 0.0 <= z >= room_d, f"z fora da sala: {z}"

        # Rotação válida
        assert -180.0 <= rot <= 180.0, f"rotação inválida: {rot}"

    print("✅ Resposta válida.")

# =========================
# TESTE PRINCIPAL
# =========================

def test_api():
    print("▶ Testando API de layout 3D...")
    print("Cena enviada:")
    print(TEST_SCENE)

    start = time.time()
    response = requests.post(API_URL, json=TEST_SCENE)
    elapsed = time.time() - start

    assert response.status_code == 200, f"Erro HTTP {response.status_code}"

    result = response.json()

    print("\n▶ Resposta recebida:")
    for i, r in enumerate(result):
        print(
            f"Objeto {i}: "
            f"x={r['x']:.2f}, "
            f"z={r['z']:.2f}, "
            f"rot={r['rotation']:.1f}°"
        )

    print(f"\n⏱ Tempo de resposta: {elapsed*1000:.2f} ms")

    validate_response(TEST_SCENE, result)

    print("\n🎉 TESTE FINALIZADO COM SUCESSO!")

# =========================
# EXECUÇÃO
# =========================

if __name__ == "__main__":
    test_api()

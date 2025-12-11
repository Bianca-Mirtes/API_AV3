# test_model_fixed.py
import json
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel

class LayoutTester:
    def __init__(self, model_path="best_layout_model.pth"):
        print("🔧 Inicializando testador de layout...")
        
        # Configurações
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📊 Dispositivo: {self.DEVICE}")
        
        # Carregar checkpoint
        print("📂 Carregando modelo treinado...")
        self.checkpoint = torch.load(model_path, map_location=self.DEVICE)
        
        # Configurações do dataset (PRIMEIRO!)
        self.obj_to_idx = self.checkpoint['obj_to_idx']
        self.max_objects = self.checkpoint.get('max_objects', 10)
        
        print(f"📋 Objetos únicos: {len(self.obj_to_idx)}")
        print(f"🎯 Máximo de objetos por layout: {self.max_objects}")
        
        # Carregar tokenizer
        self.MODEL_NAME = "distilbert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        
        # Recriar modelo (DEPOIS de ter obj_to_idx)
        print("🤖 Recriando arquitetura do modelo...")
        self.model = self.create_model()
        self.model.load_state_dict(self.checkpoint['model_state'])
        self.model.to(self.DEVICE)
        self.model.eval()
        
        print("✅ Modelo carregado com sucesso!")
    
    def create_model(self):
        """Recria a arquitetura do modelo"""
        
        # Classe interna para capturar self
        class LayoutModel(nn.Module):
            def __init__(self, tester_instance, num_obj_features):
                super().__init__()
                self.tester = tester_instance
                self.bert = AutoModel.from_pretrained(tester_instance.MODEL_NAME)
                bert_size = 768
                
                self.net = nn.Sequential(
                    nn.Linear(bert_size + num_obj_features + 2, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, tester_instance.max_objects * 3)
                )
            
            def forward(self, input_ids, attention_mask, objects_vector, room_width, room_depth):
                bert_out = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).last_hidden_state[:, 0, :]
                
                x = torch.cat([bert_out, objects_vector, room_width, room_depth], dim=1)
                return self.net(x)
        
        # Criar instância do modelo
        return LayoutModel(self, len(self.obj_to_idx))
    
    def predict(self, instruction, objects_list, room_width, room_depth):
        """Faz uma predição"""
        
        print(f"\n🧪 Predizendo layout...")
        print(f"   Instrução: {instruction}")
        print(f"   Objetos: {objects_list}")
        print(f"   Sala: {room_width}x{room_depth}")
        
        # Preparar entrada
        text = f"Instrução: {instruction} Sala: {room_width}x{room_depth} metros."
        
        # Tokenizar
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        # One-hot dos objetos
        objects_one_hot = torch.zeros(len(self.obj_to_idx), dtype=torch.float32)
        for obj in objects_list:
            if obj in self.obj_to_idx:
                objects_one_hot[self.obj_to_idx[obj]] = 1.0
            else:
                print(f"⚠️  Aviso: Objeto '{obj}' não reconhecido. Usando vetor zero.")
        
        # Converter para tensores
        input_ids = encoding['input_ids'].to(self.DEVICE)
        attention_mask = encoding['attention_mask'].to(self.DEVICE)
        objects_vector = objects_one_hot.unsqueeze(0).to(self.DEVICE)
        room_width_tensor = torch.tensor([[room_width]], dtype=torch.float32).to(self.DEVICE)
        room_depth_tensor = torch.tensor([[room_depth]], dtype=torch.float32).to(self.DEVICE)
        
        # Fazer predição
        with torch.no_grad():
            predictions = self.model(input_ids, attention_mask, objects_vector, 
                                    room_width_tensor, room_depth_tensor)
        
        # Processar resultados
        predictions = predictions.cpu().numpy()[0]
        layout = []
        
        for i, obj_name in enumerate(objects_list):
            if i >= self.max_objects:
                print(f"⚠️  {obj_name} ignorado (limite de {self.max_objects} objetos)")
                break
                
            start_idx = i * 3
            if start_idx + 2 < len(predictions):
                x = float(predictions[start_idx])
                z = float(predictions[start_idx + 1])
                rot = float(predictions[start_idx + 2])
                
                # Arredondar e limitar
                x = round(x, 2)
                z = round(z, 2)
                rot = int(round(rot))
                
                # Normalizar ângulo
                rot = rot % 360
                if rot > 180:
                    rot -= 360
                
                # Garantir limites razoáveis
                x = max(min(x, room_width/2 - 0.5), -room_width/2 + 0.5)
                z = max(min(z, room_depth/2 - 0.5), -room_depth/2 + 0.5)
                
                layout.append({
                    "name": obj_name,
                    "x": x,
                    "z": z,
                    "rot": rot
                })
                
                print(f"   📦 {obj_name}: x={x:6.2f}, z={z:6.2f}, rot={rot:4}°")
        
        return layout
    
    def test_multiple_examples(self):
        """Testa vários exemplos de uma vez"""
        
        print("\n" + "="*60)
        print("🧪 TESTANDO MÚLTIPLOS EXEMPLOS")
        print("="*60)
        
        # Mostrar objetos disponíveis
        print("\n📋 Objetos disponíveis no modelo:")
        objetos = list(self.obj_to_idx.keys())
        for i in range(0, len(objetos), 8):
            print("   " + ", ".join(objetos[i:i+8]))
        
        # Exemplos usando SOMENTE objetos que existem no modelo
        examples = [
            {
                "instruction": "quero uma sala gamer minimalista",
                "objects": ["mesa", "cadeira", "pc", "monitor"],
                "room": {"width": 4.0, "depth": 4.0}
            },
            {
                "instruction": "sala de estar aconchegante para família",
                "objects": ["sofa", "mesa_de_centro", "televisao", "estante", "tapete"],
                "room": {"width": 5.0, "depth": 6.0}
            },
            {
                "instruction": "escritório funcional para home office",
                "objects": ["mesa", "cadeira", "estante", "luminaria"],
                "room": {"width": 3.5, "depth": 4.0}
            },
            {
                "instruction": "quarto compacto para estudante",
                "objects": ["cama", "guarda_roupa", "mesa", "cadeira"],
                "room": {"width": 3.0, "depth": 4.0}
            },
            {
                "instruction": "cozinha americana prática",
                "objects": ["mesa", "cadeira", "pia", "armario"],
                "room": {"width": 4.0, "depth": 5.0}
            }
        ]
        
        # Filtrar para usar apenas objetos existentes
        filtered_examples = []
        for ex in examples:
            valid_objs = [obj for obj in ex["objects"] if obj in self.obj_to_idx]
            if valid_objs:
                filtered_ex = ex.copy()
                filtered_ex["objects"] = valid_objs
                filtered_examples.append(filtered_ex)
                if len(valid_objs) < len(ex["objects"]):
                    invalid = [obj for obj in ex["objects"] if obj not in self.obj_to_idx]
                    print(f"⚠️  Exemplo '{ex['instruction'][:30]}...': removidos objetos inválidos: {invalid}")
        
        all_results = []
        
        for i, example in enumerate(filtered_examples, 1):
            print(f"\n📝 Exemplo {i}:")
            print("-" * 40)
            
            layout = self.predict(
                example["instruction"],
                example["objects"],
                example["room"]["width"],
                example["room"]["depth"]
            )
            
            # Salvar resultado
            result = {
                "instruction": example["instruction"],
                "objects": example["objects"],
                "room": example["room"],
                "output": layout
            }
            
            all_results.append(result)
            
            # Salvar individualmente
            with open(f"layout_exemplo_{i}.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Salvo em 'layout_exemplo_{i}.json'")
        
        # Salvar todos os resultados
        if all_results:
            with open("todos_layouts_testados.json", "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ Todos os resultados salvos em 'todos_layouts_testados.json'")
        else:
            print("\n⚠️  Nenhum exemplo válido para testar!")
        
        return all_results

# ==================== FUNÇÃO PRINCIPAL ====================
def main():
    print("="*60)
    print("🔬 TESTADOR DE MODELO DE LAYOUT 3D")
    print("="*60)
    
    try:
        # Inicializar testador
        tester = LayoutTester()
        
        # Testar
        tester.test_multiple_examples()
        
        print("\n" + "="*60)
        print("✅ TODOS OS TESTES CONCLUÍDOS!")
        print("="*60)
        
        # Mostrar estatísticas
        print("\n📊 Estatísticas do modelo:")
        print(f"   - Objetos reconhecidos: {len(tester.obj_to_idx)}")
        print(f"   - Máximo de objetos por layout: {tester.max_objects}")
        print(f"   - Dispositivo utilizado: {tester.DEVICE}")
        
        # Mostrar exemplo dos objetos
        print(f"\n📝 Exemplo de objetos: {list(tester.obj_to_idx.keys())[:10]}...")
        
    except FileNotFoundError:
        print("❌ ERRO: Modelo não encontrado!")
        print("   Certifique-se de que 'best_layout_model.pth' existe")
        print("   Ou tente: LayoutTester('fast_layout_model.pth')")
    except KeyError as e:
        print(f"❌ ERRO: Checkpoint corrompido ou formato inválido!")
        print(f"   Campo faltando: {e}")
        print("   Verifique se o arquivo .pth foi gerado corretamente")
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()

# ==================== EXECUTAR ====================
if __name__ == "__main__":
    main()
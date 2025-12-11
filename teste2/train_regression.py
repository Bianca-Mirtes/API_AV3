# train_fast.py
import json
import torch
import torch.nn as nn
import time
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("⚡ VERSÃO OTIMIZADA ULTRA-RÁPIDA")
print("=" * 60)

# ==================== TIMER ====================
class Timer:
    def __init__(self, name):
        self.name = name
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        elapsed = time.time() - self.start
        print(f"⏱️  {self.name}: {elapsed:.2f} segundos")

# ==================== CONFIG ====================
MODEL_NAME = "distilbert-base-uncased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"📊 Dispositivo: {DEVICE}")
print(f"🤖 Modelo: {MODEL_NAME}")

# ==================== DATASET OTIMIZADO ====================
class FastLayoutDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_objects=8):
        print("📂 Lendo dataset...")
        with Timer("  Leitura do arquivo"):
            with open(filepath, 'r', encoding='utf-8') as f:
                self.data = [json.loads(line) for line in f]
        
        print(f"📈 Total: {len(self.data)} amostras")
        
        # Coletar objetos únicos RÁPIDO
        with Timer("  Processamento de objetos"):
            all_objects = set()
            for item in self.data:
                all_objects.update(item['objects'])
            
            self.obj_to_idx = {obj: i for i, obj in enumerate(sorted(all_objects))}
            print(f"📋 Objetos únicos: {len(all_objects)}")
        
        # PRÉ-COMPUTAR TUDO
        with Timer("  Pré-computação de features"):
            self.max_objects = max_objects
            self.max_targets = max_objects * 3
            
            # Pré-processar TUDO de uma vez
            self.cache = []
            
            for item in self.data:
                # Texto simplificado
                text = f"{item['instruction']} {item['room']['width']}x{item['room']['depth']}"
                
                # Tokenizar AGORA (não no __getitem__)
                encoding = tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=64,  # REDUZIDO
                    return_tensors='pt'
                )
                
                # One-hot encoding
                obj_vector = torch.zeros(len(self.obj_to_idx), dtype=torch.float32)
                for obj in item['objects']:
                    idx = self.obj_to_idx.get(obj)
                    if idx is not None:
                        obj_vector[idx] = 1.0
                
                # Targets
                targets = torch.zeros(self.max_targets, dtype=torch.float32)
                
                # Mapear output
                output_map = {o['name']: o for o in item['output']}
                
                for i, obj_name in enumerate(item['objects'][:max_objects]):
                    if obj_name in output_map:
                        obj_data = output_map[obj_name]
                        idx = i * 3
                        targets[idx] = obj_data['x']
                        targets[idx + 1] = obj_data['z']
                        targets[idx + 2] = obj_data['rot']
                
                # Room features
                room_width = torch.tensor([item['room']['width']], dtype=torch.float32)
                room_depth = torch.tensor([item['room']['depth']], dtype=torch.float32)
                
                # Guardar no cache
                self.cache.append({
                    'input_ids': encoding['input_ids'].squeeze(0),
                    'attention_mask': encoding['attention_mask'].squeeze(0),
                    'objects_vector': obj_vector,
                    'room_width': room_width,
                    'room_depth': room_depth,
                    'targets': targets,
                    'num_objects': min(len(item['objects']), max_objects)
                })
        
        print(f"✅ Dataset pré-processado: {len(self.cache)} itens em cache")
    
    def __len__(self):
        return len(self.cache)
    
    def __getitem__(self, idx):
        # Retorna do cache (SUPER RÁPIDO)
        return self.cache[idx]

# ==================== MODELO LEVE ====================
class FastLayoutModel(nn.Module):
    def __init__(self, num_obj_features):
        super().__init__()
        
        # BERT pequeno
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        bert_size = 768  # DistilBERT
        
        # MLP eficiente
        self.net = nn.Sequential(
            nn.Linear(bert_size + num_obj_features + 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8 * 3)  # 8 objetos * 3 valores
        )
    
    def forward(self, input_ids, attention_mask, objects_vector, room_width, room_depth):
        # BERT
        with torch.no_grad():  # Freeze BERT por enquanto
            bert_out = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).last_hidden_state[:, 0, :]
        
        # Concatenar
        x = torch.cat([bert_out, objects_vector, room_width, room_depth], dim=1)
        
        # MLP
        return self.net(x)

# ==================== MAIN OTIMIZADO ====================
def main():
    print("\n🚀 INICIANDO PROCESSAMENTO...")
    
    # 1. Tokenizer
    with Timer("Carregar tokenizer"):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 2. Dataset
    with Timer("Carregar e pré-processar dataset"):
        dataset = FastLayoutDataset("dataset.jsonl", tokenizer, max_objects=8)
    
    # 3. Split rápido
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    with Timer("Split dataset"):
        train_set, val_set = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
    
    print(f"📊 Treino: {len(train_set)}, Validação: {len(val_set)}")
    
    # 4. DataLoaders
    with Timer("Criar DataLoaders"):
        train_loader = DataLoader(
            train_set,
            batch_size=16 if DEVICE == "cuda" else 8,
            shuffle=True,
            num_workers=0,
            pin_memory=True if DEVICE == "cuda" else False
        )
        
        val_loader = DataLoader(
            val_set,
            batch_size=16 if DEVICE == "cuda" else 8,
            shuffle=False,
            num_workers=0
        )
    
    # 5. Modelo
    with Timer("Carregar modelo"):
        model = FastLayoutModel(len(dataset.obj_to_idx))
        
        if DEVICE == "cuda":
            model.cuda()
            print("✅ Modelo na GPU")
        
        # Congelar BERT inicialmente (treina mais rápido)
        for param in model.bert.parameters():
            param.requires_grad = False
    
    # 6. Otimizador
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3
    )
    
    criterion = nn.MSELoss()
    
    # 7. TREINAMENTO RÁPIDO
    print("\n🎯 INICIANDO TREINAMENTO RÁPIDO")
    print("=" * 60)
    
    for epoch in range(5):  # Apenas 5 épocas para teste
        with Timer(f"Época {epoch + 1}"):
            # Treino
            model.train()
            train_loss = 0.0
            batches = 0
            
            for batch in train_loader:
                # Mover para GPU
                if DEVICE == "cuda":
                    batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
                
                # Forward
                pred = model(
                    batch['input_ids'],
                    batch['attention_mask'],
                    batch['objects_vector'],
                    batch['room_width'],
                    batch['room_depth']
                )
                
                loss = criterion(pred, batch['targets'])
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                batches += 1
            
            # Validação
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    if DEVICE == "cuda":
                        batch = {k: v.cuda() for k, v in batch.items()}
                    
                    pred = model(
                        batch['input_ids'],
                        batch['attention_mask'],
                        batch['objects_vector'],
                        batch['room_width'],
                        batch['room_depth']
                    )
                    
                    loss = criterion(pred, batch['targets'])
                    val_loss += loss.item()
                    val_batches += 1
            
            # Stats
            avg_train = train_loss / batches if batches > 0 else 0
            avg_val = val_loss / val_batches if val_batches > 0 else 0
            
            print(f"Época {epoch + 1}: Treino={avg_train:.4f}, Val={avg_val:.4f}")
    
    # 8. Salvar
    with Timer("Salvar modelo"):
        torch.save({
            'model_state': model.state_dict(),
            'obj_to_idx': dataset.obj_to_idx,
            'config': {
                'model_name': MODEL_NAME,
                'max_objects': 8
            }
        }, "best_layout_model.pth")
    
    print("\n✅ TREINAMENTO COMPLETO!")

# ==================== TESTE INSTANTÂNEO ====================
def quick_test():
    """Teste rápido sem treinar"""
    
    print("\n🔍 TESTE RÁPIDO DE PERFORMANCE")
    print("=" * 60)
    
    # Medir tempo de tokenização
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    test_text = "quero uma sala gamer minimalista"
    
    with Timer("Tokenizar 1000x"):
        for _ in range(1000):
            tokenizer(test_text, return_tensors='pt')
    
    # Medir tempo de carregamento de modelo
    with Timer("Carregar modelo BERT"):
        model = AutoModel.from_pretrained(MODEL_NAME)
    
    # Verificar GPU
    if torch.cuda.is_available():
        print(f"\n🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"📊 VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Teste simples na GPU
        with Timer("Inferência na GPU"):
            model.cuda()
            dummy_input = torch.randint(0, 1000, (1, 32)).cuda()
            with torch.no_grad():
                _ = model(dummy_input)
    
    print("\n✅ Teste de performance completo")

# ==================== EXECUTAR ====================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        quick_test()
    else:
        main()
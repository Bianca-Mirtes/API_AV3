# train_optimized_12gb.py
import json
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# CONFIGURAÇÕES CRÍTICAS para economizar VRAM
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():
    print("=" * 60)
    print("🚀 TREINAMENTO OTIMIZADO PARA RTX 3060 12GB")
    print("=" * 60)
    
    # 1. LIMPAR MEMÓRIA ANTES DE COMEÇAR
    torch.cuda.empty_cache()
    gc.collect()
    
    # 2. MODELO LEVE - DistilGPT-2 é bom para começar
    model_name = "distilgpt2"
    print(f"📊 Modelo: {model_name}")
    
    # 3. CARREGAR COM CONFIGURAÇÕES DE ECONOMIA DE MEMÓRIA
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("🤖 Carregando modelo...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Usar half precision
    ).to("cuda")
    
    # 4. HABILITAR GRADIENT CHECKPOINTING (economiza VRAM)
    model.gradient_checkpointing_enable()
    
    # 5. CARREGAR DADOS EM BATCHES PEQUENOS
    print("📂 Carregando dados...")
    
    with open("dataset.jsonl", "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    print(f"📈 Total de amostras: {len(lines)}")
    
    # 6. FUNÇÃO PARA PROCESSAR EM MINI-BATCHES
    def process_in_mini_batches(lines, batch_size=2, max_samples=100):
        """Processa dados em mini-batches para economizar memória"""
        processed = []
        
        # Limitar a um número gerenciável inicialmente
        lines = lines[:max_samples] if max_samples else lines
        
        print(f"🔄 Processando {len(lines)} amostras em batches de {batch_size}...")
        
        for i in range(0, len(lines), batch_size):
            batch_lines = lines[i:i+batch_size]
            batch_texts = []
            
            for line in batch_lines:
                item = json.loads(line)
                text = (
                    f"Instruction: {item['instruction']}\n"
                    f"Objects: {', '.join(item['objects'])}\n"
                    f"Room: {item['room']['width']}x{item['room']['depth']}\n"
                    f"Output: {json.dumps(item['output'])}"
                )
                batch_texts.append(text)
            
            # Tokenizar este mini-batch
            batch_tokens = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=192,  # REDUZIDO para economizar memória
                return_tensors="pt"
            ).to("cuda")
            
            processed.append(batch_tokens)
            
            # Limpar memória entre batches
            if i % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                print(f"  Processados {i+len(batch_lines)}/{len(lines)} amostras")
        
        return processed, len(lines)
    
    # 7. PROCESSAR DADOS
    tokenized_batches, total_samples = process_in_mini_batches(
        lines, 
        batch_size=2,  # BATCH PEQUENO
        max_samples=200  # Comece com pouco, depois aumente
    )
    
    print(f"✅ Dados preparados: {len(tokenized_batches)} batches, {total_samples} amostras")
    
    # 8. CONFIGURAR TREINAMENTO COM GRADIENT ACCUMULATION
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    
    # 9. LOOP DE TREINAMENTO COM GERENCIAMENTO DE MEMÓRIA
    print("\n🎯 Iniciando treinamento...")
    print("-" * 40)
    
    gradient_accumulation_steps = 8  # Acumula gradientes para batch efetivo maior
    model.train()
    
    for epoch in range(3):  # Comece com poucas épocas
        print(f"\n📅 ÉPOCA {epoch + 1}/3")
        print("-" * 30)
        
        total_loss = 0
        optimizer.zero_grad()  # Resetar gradientes no início da época
        
        for batch_idx, batch in enumerate(tokenized_batches):
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["input_ids"]
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward com escala de loss para accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # Atualizar pesos apenas a cada accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                # Limpar memória
                torch.cuda.empty_cache()
                gc.collect()
            
            # Log progresso
            if (batch_idx + 1) % 5 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"  Batch {batch_idx + 1}/{len(tokenized_batches)} - Loss: {loss.item()*gradient_accumulation_steps:.4f} (média: {avg_loss:.4f})")
        
        # Última atualização se necessário
        if len(tokenized_batches) % gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Estatísticas da época
        avg_epoch_loss = total_loss / len(tokenized_batches)
        print(f"📊 Época {epoch + 1} concluída - Loss médio: {avg_epoch_loss:.4f}")
        
        # Salvar checkpoint a cada época
        checkpoint_path = f"./checkpoints/epoch_{epoch+1}"
        os.makedirs(checkpoint_path, exist_ok=True)
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        print(f"💾 Checkpoint salvo: {checkpoint_path}")
        
        # Limpar memória entre épocas
        torch.cuda.empty_cache()
        gc.collect()
    
    # 10. SALVAR MODELO FINAL
    print("\n" + "=" * 60)
    print("💾 Salvando modelo final...")
    
    final_path = "./layout_model_final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    print(f"✅ Modelo salvo em: {final_path}")
    print("=" * 60)
    
    # 11. VERIFICAR USO DE MEMÓRIA
    print("\n📊 Estatísticas finais de memória:")
    print(f"  VRAM alocada: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"  VRAM reservada: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print(f"  VRAM máxima alocada: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

if __name__ == "__main__":
    main()
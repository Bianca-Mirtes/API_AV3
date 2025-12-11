# train_model_properly.py
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import os

# Configurar para evitar problemas no Windows
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def create_formatted_examples():
    """Cria exemplos formatados corretamente para o modelo aprender"""
    data = []
    
    with open("training_dataset.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            
            # FORMATO CRÍTICO: Igual ao que queremos na inferência
            text = f"""Instruction: {item['instruction']}
Objects: {json.dumps(item['objects'])}
Room: {json.dumps(item['room'])}
Output: {json.dumps(item['output'])}"""
            
            data.append({"text": text})
    
    return data

def main():
    print("🧠 TREINAMENTO DO MODELO DE LAYOUT")
    print("=" * 50)
    
    # 1. Carregar dataset formatado
    print("📂 Carregando dataset...")
    formatted_data = create_formatted_examples()
    dataset = Dataset.from_list(formatted_data)
    
    print(f"📊 Dataset: {len(dataset)} exemplos")
    
    # 2. Carregar modelo e tokenizer
    MODEL_NAME = "distilgpt2"  # Manha com este, depois podemos usar maior
    
    print(f"🤖 Carregando {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # CRÍTICO: Configurar tokenizer corretamente
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 3. Tokenizar
    print("🔠 Tokenizando dataset...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors=None
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=4,
        remove_columns=["text"]
    )
    
    # 4. Carregar modelo
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    # 5. Configurar treinamento com MAIS ÉPOCAS
    training_args = TrainingArguments(
        output_dir="./layout-model-trained",
        overwrite_output_dir=True,
        num_train_epochs=20,  # AUMENTE isto! Comece com 20
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,  # Para batch efetivo de 16
        warmup_steps=100,
        logging_steps=50,
        save_steps=500,
        learning_rate=3e-5,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,  # Economiza memória
        optim="adamw_torch",
        report_to="none",
        dataloader_num_workers=0,  # 0 para Windows
        save_total_limit=2,
        no_cuda=not torch.cuda.is_available(),
    )
    
    # 6. Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 7. Treinar
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("🎯 Iniciando treinamento...")
    print(f"📈 Épocas: {training_args.num_train_epochs}")
    print(f"📊 Batch size efetivo: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print("=" * 50)
    
    trainer.train()
    
    # 8. Salvar
    print("💾 Salvando modelo...")
    model.save_pretrained("./layout-model-final-trained")
    tokenizer.save_pretrained("./layout-model-final-trained")
    
    print("✅ Treinamento concluído!")
    
    # 9. Testar o modelo treinado
    test_trained_model()

def test_trained_model():
    """Testa o modelo logo após treinamento"""
    print("\n🧪 TESTANDO MODELO TREINADO")
    print("=" * 50)
    
    # Carregar modelo treinado
    tokenizer = AutoTokenizer.from_pretrained("./layout-model-final-trained")
    model = AutoModelForCausalLM.from_pretrained("./layout-model-final-trained")
    
    if torch.cuda.is_available():
        model.cuda()
    
    # Testar com exemplos do treino
    test_examples = [
        {
            "instruction": "crie uma sala gamer minimalista",
            "objects": ["mesa", "cadeira", "pc"],
            "room": {"width": 4.0, "depth": 4.0}
        },
        {
            "instruction": "organize uma sala de estar",
            "objects": ["sofa", "tv", "mesa_de_centro"],
            "room": {"width": 5.0, "depth": 6.0}
        }
    ]
    
    for i, example in enumerate(test_examples):
        print(f"\n📊 Teste {i+1}: {example['instruction']}")
        
        prompt = f"""Instruction: {example['instruction']}
Objects: {json.dumps(example['objects'])}
Room: {json.dumps(example['room'])}
Output:"""
        
        print(f"📝 Prompt: {prompt[:100]}...")
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extrair apenas a parte após Output:
        if "Output:" in generated:
            output_part = generated.split("Output:")[-1].strip()
            print(f"🎯 Saída gerada: {output_part}")
            
            # Verificar se parece JSON
            if "[" in output_part and "]" in output_part:
                print("✅ PARECE BOM! Gerando JSON...")
                try:
                    # Tentar extrair JSON
                    import re
                    json_match = re.search(r'\[.*\]', output_part, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        parsed = json.loads(json_str)
                        print(f"✅ JSON válido com {len(parsed)} objetos!")
                        for obj in parsed:
                            print(f"   {obj.get('name', 'unnamed')}: ({obj.get('x', 0)}, {obj.get('z', 0)})")
                except:
                    print("⚠️  JSON mal-formado, mas pelo menos tentou!")
            else:
                print("❌ Não gerou JSON - precisa de mais treino")
        else:
            print("❌ Nem gerou 'Output:' - precisa de MUITO mais treino")

if __name__ == "__main__":
    main()
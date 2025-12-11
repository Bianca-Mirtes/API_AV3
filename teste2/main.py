# main_fixed_body.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import json
import uuid
import base64
import os
import aiofiles
import whisper  # OpenAI Whisper
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn
from typing import List
import tempfile
import re

app = FastAPI(title="Audio → Layout 3D Generator")
TMP = "tmp"
os.makedirs(TMP, exist_ok=True)

# Configurar dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📱 Usando dispositivo: {device}")

# Carregar modelo STT (OpenAI Whisper)
print("🎤 Carregando modelo Whisper...")
stt_model = whisper.load_model("base").to(device)

print("🤖 Carregando modelo de layout 3D...")

# Tentar carregar o modelo treinado, se falhar usar fallback
MODEL_PATH = "./layout_model_final"

try:
    # Primeiro verificar se o diretório existe
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  Diretório {MODEL_PATH} não encontrado. Usando fallback...")
        raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}")
    
    # Tentar carregar o tokenizer primeiro
    print("📝 Carregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Configurar tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("🧠 Carregando modelo...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True
    )
    
    if device == "cuda":
        model.to(device)
    
    model.eval()
    print("✅ Modelo de layout carregado com sucesso!")
    
except Exception as e:
    print(f"⚠️  Erro ao carregar modelo treinado: {e}")
    print("🔄 Usando DistilGPT-2 como fallback...")
    
    # Fallback para modelo padrão
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    
    if device == "cuda":
        model.cuda()
    
    model.eval()

# Schemas
class RoomDimensions(BaseModel):
    width: float
    depth: float

class Position(BaseModel):
    name: str
    x: float
    y: float = 0.0
    z: float
    rot: float

class Audio3DRequest(BaseModel):
    audio_base64: str
    objects: List[str]
    room: RoomDimensions

class Layout3DResponse(BaseModel):
    instruction: str
    objects: List[str]
    room: RoomDimensions
    layout: List[Position]
    audio_id: str

# NOVO: Schema para requisição de texto
class TextLayoutRequest(BaseModel):
    instruction: str
    objects: List[str]
    width: float
    depth: float

def extract_json_from_text(text: str):
    """Extrai JSON de texto gerado pelo modelo"""
    # Tentar encontrar array JSON
    pattern = r'\[(\s*\{.*?\}\s*,?\s*)+\]'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        try:
            return json.loads(matches[0])
        except:
            pass
    
    # Tentar encontrar objetos individuais
    pattern = r'\{[^{}]*\}'
    matches = re.findall(pattern, text)
    
    objects = []
    for match in matches:
        try:
            obj = json.loads(match)
            if 'name' in obj and 'x' in obj:
                objects.append(obj)
        except:
            continue
    
    return objects if objects else None

# Função de normalização dos nomes dos objetos
def normalize_object_name(obj_name: str) -> str:
    """Normaliza nomes de objetos para os nomes simples usados no treinamento"""
    obj_name = obj_name.lower().strip()
    
    # Mapeamento de sinônimos/variações para nomes padrão
    object_mapping = {
        # Sofás
        "sofa": "sofa",
        "sofá": "sofa",
        "sofa_de_couro": "sofa",
        "sofá_de_couro": "sofa",
        "sofa_de_tecido": "sofa",
        "sofá_de_tecido": "sofa",
        "couch": "sofa",
        "sofá_retrátil": "sofa",
        
        # Mesas
        "mesa": "mesa",
        "mesa_de_centro": "mesa_de_centro",
        "mesa_de_canto": "mesa",
        "mesa_auxiliar": "mesa",
        "mesa_de_jantar": "mesa",
        "mesa_gamer": "mesa",
        
        # Cadeiras
        "cadeira": "cadeira",
        "cadeira_gamer": "cadeira",
        "cadeira_de_escritório": "cadeira",
        "cadeira_de_balanço": "cadeira",
        "poltrona": "cadeira",
        "cadeira_de_jantar": "cadeira",
        
        # Camas
        "cama": "cama",
        "cama_de_casal": "cama",
        "cama_de_solteiro": "cama",
        "cama_de_quarto": "cama",
        "cama_king_size": "cama",
        "cama_box": "cama",
        
        # Armários/Estantes
        "estante": "estante",
        "armário": "armario",
        "armario": "armario",
        "guarda_roupa": "guarda_roupa",
        "rack": "rack",
        "rack_de_tv": "rack",
        "prateleira": "estante",
        
        # Eletrônicos
        "pc": "pc",
        "computador": "pc",
        "monitor": "monitor",
        "tv": "tv",
        "televisão": "tv",
        "televisao": "tv",
        
        # Tapetes
        "tapete": "tapete",
        "tapete_persa": "tapete",
        "carpete": "tapete",
        
        # Cozinha/Banheiro
        "pia": "pia",
        "pia_de_banheiro": "pia",
        "pia_de_cozinha": "pia",
        "geladeira": "geladeira",
        "fogão": "fogao",
        "fogao": "fogao",
        "luminária": "luminaria",
        "luminaria": "luminaria",
        
        # Outros
        "espelho": "espelho",
        "quadro": "quadro",
        "vaso": "vaso",
        "planta": "planta",
    }
    
    # Verificar mapeamento exato
    if obj_name in object_mapping:
        return object_mapping[obj_name]
    
    # Tentar encontrar por substring
    for key in object_mapping.keys():
        if key in obj_name and len(key) > 2:  # Evitar falsos positivos com palavras curtas
            return object_mapping[key]
    
    # Se não encontrar, tentar extrair a palavra principal
    # Remove sufixos comuns
    words_to_remove = ["de_", "do_", "da_", "com_", "sem_", "_de_", "_do_", "_da_"]
    
    simple_name = obj_name
    for word in words_to_remove:
        simple_name = simple_name.replace(word, "_")
    
    # Pega a primeira palavra separada por "_"
    parts = simple_name.split("_")
    if parts:
        return parts[0]
    
    return obj_name  # Retorna original se nada funcionar

# Usar na função generate_layout, ANTES de processar:
def normalize_object_list(objects: List[str]) -> List[str]:
    """Normaliza uma lista de nomes de objetos"""
    normalized = []
    for obj in objects:
        norm_obj = normalize_object_name(obj)
        normalized.append(norm_obj)
        if obj != norm_obj:
            print(f"   🔄 Normalizado: '{obj}' → '{norm_obj}'")
    return normalized

async def generate_layout(instruction: str, objects: List[str], room_width: float, room_depth: float):
    """Gera layout 3D baseado na instrução e objetos"""
    
    try:
        # NORMALIZAR NOMES DOS OBJETOS
        normalized_objects = normalize_object_list(objects)

        # Preparar prompt no formato esperado pelo modelo
        prompt = f"Instruction: {instruction}\nObjects: {', '.join(objects)}\nRoom: {room_width}x{room_depth}\nOutput:"
        prompt = f"""Instruction: {instruction}
            Objects: {json.dumps(objects)}
            Room: {json.dumps({"width": room_width, "depth": room_depth})}
            Output: """
        
        print(f"🎯 Prompt enviado ao modelo:")
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
        
        # Tokenizar
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False
        )
        
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Gerar com configurações otimizadas
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=350,  # Aumentado
                temperature=0.8,  # Mais criativo
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.2,  # Evitar repetição
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,  # Evitar repetição de n-grams
                early_stopping=True
            )
        
        # Decodificar
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"📄 Texto gerado pelo modelo: {generated_text}")
        
        # Extrair apenas a parte APÓS "Output:"
        if "Output:" in generated_text:
            output_part = generated_text.split("Output:")[-1].strip()
            print(f"🎯 Parte após 'Output:': {output_part}")
            
            # Tentar extrair JSON
            try:
                # Limpar texto - remover espaços extras e newlines
                clean_output = output_part.replace('\n', ' ').replace('\r', '')
                
                # Procurar por array JSON
                import re
                json_pattern = r'\[\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*\]'
                match = re.search(json_pattern, clean_output, re.DOTALL)
                
                if match:
                    json_str = match.group(0)
                    print(f"🎯 JSON encontrado: {json_str[:100]}...")
                    
                    # Converter para objetos Python
                    positions_data = json.loads(json_str)
                    
                    # Converter para objetos Position
                    positions = []
                    for i, item in enumerate(positions_data):
                        # Se tiver menos itens que objetos, usar nome do objeto
                        if i < len(objects):
                            obj_name = objects[i]
                        else:
                            obj_name = item.get('name', f'object_{i}')
                        
                        positions.append(Position(
                            name=obj_name,
                            x=float(item.get('x', 0.0)),
                            z=float(item.get('z', 0.0)),
                            rot=float(item.get('rot', 0.0)),
                            y=float(item.get('y', 0.0))
                        ))
                    
                    return positions
                else:
                    print("⚠️  Não encontrou padrão JSON na saída")
                    
            except json.JSONDecodeError as e:
                print(f"❌ Erro ao decodificar JSON: {e}")
                print(f"📝 Texto que causou erro: {clean_output[:200]}...")

    except Exception as e:
        print(f"❌ Erro na geração do layout: {e}")

@app.post("/generate-layout", response_model=Layout3DResponse)
async def generate_3d_layout(request: Audio3DRequest):
    """
    Endpoint principal: Recebe áudio com instrução, lista de objetos e dimensões da sala.
    Retorna layout 3D com posições dos objetos.
    
    Body esperado:
    {
        "audio_base64": "base64_string",
        "objects": ["mesa", "cadeira", "pc"],
        "room": {
            "width": 4.0,
            "depth": 4.0
        }
    }
    """
    try:
        # Gerar ID único para este request
        audio_id = str(uuid.uuid4())
        audio_path = f"{TMP}/{audio_id}.wav"
        
        print(f"🎯 Processando requisição {audio_id}")
        print(f"   Objetos: {request.objects}")
        print(f"   Sala: {request.room.width}x{request.room.depth}")
        
        # 1. Salvar áudio temporário
        audio_bytes = base64.b64decode(request.audio_base64)
        async with aiofiles.open(audio_path, "wb") as f:
            await f.write(audio_bytes)
        
        # 2. STT → Transcrição (OpenAI Whisper)
        print("   🔊 Transcrevendo áudio...")
        # Criar arquivo temporário em memória para o Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            temp_audio_path = tmp.name
        
        try:
            # Transcrever usando o arquivo temporário
            result = stt_model.transcribe(temp_audio_path, language="pt")
            instruction = result["text"].strip()
        except Exception as e:
            print(f"   ⚠️  Erro no Whisper: {e}")
            instruction = "Organize os objetos de forma funcional e harmoniosa"
        
        if not instruction or len(instruction) < 3:
            instruction = "Organize os objetos de forma funcional e harmoniosa"
            print("   ⚠️  Transcrição vazia ou muito curta, usando instrução padrão")
        
        print(f"   📝 Instrução: {instruction}")
        
        # 3. Gerar layout 3D
        print("   🎨 Gerando layout...")
        layout = await generate_layout(
            instruction=instruction,
            objects=request.objects,
            room_width=request.room.width,
            room_depth=request.room.depth
        )
        
        print(f"   ✅ Layout gerado com {len(layout)} posições")
        
        # 4. Limpar arquivo temporário
        try:
            os.remove(audio_path)
        except:
            pass
        
        # 5. Retornar resposta
        return Layout3DResponse(
            instruction=instruction,
            objects=request.objects,
            room=request.room,
            layout=layout,
            audio_id=audio_id
        )
        
    except Exception as e:
        print(f"❌ Erro no processamento: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.post("/generate-layout-from-text", response_model=Layout3DResponse)
async def generate_layout_from_text(request: TextLayoutRequest):
    """
    Endpoint alternativo para testar sem áudio.
    
    Body esperado:
    {
        "instruction": "criar uma sala gamer minimalista",
        "objects": ["mesa", "cadeira", "pc"],
        "width": 4.0,
        "depth": 4.0
    }
    """
    try:
        print(f"🎯 Gerando layout a partir de texto")
        print(f"   Instrução: {request.instruction}")
        print(f"   Objetos: {request.objects}")
        print(f"   Sala: {request.width}x{request.depth}")
        
        layout = await generate_layout(
            instruction=request.instruction,
            objects=request.objects,
            room_width=request.width,
            room_depth=request.depth
        )
        
        return Layout3DResponse(
            instruction=request.instruction,
            objects=request.objects,
            room=RoomDimensions(width=request.width, depth=request.depth),
            layout=layout,
            audio_id=str(uuid.uuid4())
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Endpoint de verificação de saúde"""
    return {
        "status": "healthy",
        "device": device,
        "model_loaded": model is not None,
        "stt_model": "whisper-base",
        "service": "Layout 3D Generator API"
    }

def test_model_training():
    """Testa se o modelo aprendeu a tarefa"""
    print("🧪 TESTANDO QUALIDADE DO TREINAMENTO")
    print("=" * 50)
    
    # Usar exemplo do dataset de treino
    test_examples = [
        {
            "instruction": "organize um sala de jogos para mim",
            "objects": ["guarda_roupa", "mesa", "armario", "luminaria"],
            "width": 7.3,
            "depth": 4.7
        },
        {
            "instruction": "criar uma sala gamer",
            "objects": ["mesa", "cadeira", "pc"],
            "width": 4.0,
            "depth": 4.0
        }
    ]
    
    for i, example in enumerate(test_examples):
        print(f"\n📊 Teste {i+1}: {example['instruction']}")
        
        prompt = f"""Instruction: {example['instruction']}
            Objects: {json.dumps(example['objects'])}
            Room: {json.dumps({"width": example['width'], "depth": example['depth']})}
            Output: """
        
        print(f"📝 Prompt: {prompt[:100]}...")
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"🎯 Resposta: {generated}")
        
        if "Output:" in generated:
            output_part = generated.split("Output:")[-1].strip()
            print(f"🔍 Após 'Output:': {output_part[:100]}...")
            
            if "[" in output_part and "]" in output_part:
                print("✅ Parece estar gerando JSON!")
            else:
                print("❌ Não está gerando JSON - modelo pode não estar treinado")
        else:
            print("❌ Nem sequer gerou 'Output:'")
    
    print("\n" + "=" * 50)
    print("CONCLUSÃO: Se não estiver gerando JSON, o modelo precisa de mais treinamento!")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 API Layout 3D Generator")
    print("="*60)
    print(f"📁 Diretório temporário: {TMP}")
    print(f"🎮 GPU disponível: {torch.cuda.is_available()}")
    print(f"🔧 Dispositivo: {device}")
    print(f"🤖 Modelo carregado: {model is not None}")
    print("="*60 + "\n")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8081,
        log_level="info"
    )
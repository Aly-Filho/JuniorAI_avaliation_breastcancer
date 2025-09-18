import numpy as np
from PIL import Image
import os
import medmnist
from medmnist import INFO

# --- Configuração ---
DATA_NAME = 'breastmnist'
# Criar uma pasta para guardar as imagens de teste, se ela não existir
OUTPUT_FOLDER = 'imagens_de_teste'
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Número de imagens que você quer extrair
NUM_IMAGENS_PARA_EXTRAIR = 20

# --- Lógica ---
print(f"A descarregar o dataset '{DATA_NAME}' para encontrar o ficheiro...")

# A biblioteca vai descarregar os dados (ou usar o cache se já os tiver)
info = INFO[DATA_NAME]
data = getattr(medmnist, info['python_class'])(split='test', download=True)

# Carregar as imagens e as etiquetas (labels) do dataset de teste
imagens = data.imgs
etiquetas = data.labels.flatten() # Achatamos o array para ser mais fácil de usar

print(f"A extrair as primeiras {NUM_IMAGENS_PARA_EXTRAIR} imagens para a pasta '{OUTPUT_FOLDER}'...")

# Iterar sobre o número de imagens que queremos guardar
for i in range(NUM_IMAGENS_PARA_EXTRAIR):
    # Obter uma imagem e a sua etiqueta
    imagem_array = imagens[i]
    etiqueta = etiquetas[i]
    
    # Mapear a etiqueta para um nome compreensível (0: Benigno, 1: Maligno)
    nome_classe = "benigno" if etiqueta == 0 else "maligno"
    
    # Criar a imagem a partir do array numpy
    imagem_pil = Image.fromarray(imagem_array)
    
    # Gerar um nome de ficheiro descritivo
    nome_ficheiro = f"imagem_{i+1}_diagnostico_{nome_classe}.png"
    caminho_ficheiro = os.path.join(OUTPUT_FOLDER, nome_ficheiro)
    
    # Guardar a imagem
    imagem_pil.save(caminho_ficheiro)

print("Extração concluída com sucesso!")
print(f"Pode agora usar as imagens na pasta '{OUTPUT_FOLDER}' para testar a sua aplicação.")
import torch
import torch.nn as nn
from torchvision import transforms
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
import io

# --- 1. Definição do Modelo ---
# É CRUCIAL ter a mesma arquitetura de modelo que foi usada no treino.
# Copie a classe do modelo do seu script de treino para aqui.
class SimpleCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SimpleCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1) # Flatten the tensor
        out = self.fc(out)
        return out

# --- 2. Carregar o Modelo Treinado ---
# Inicializar o modelo e carregar os pesos guardados.
MODEL_PATH = 'breast_mnist_model.pth'
N_CHANNELS = 1
N_CLASSES = 2
CLASS_NAMES = {0: 'Benigno', 1: 'Maligno'} # 0 -> benign, 1 -> malignant

model = SimpleCNN(in_channels=N_CHANNELS, num_classes=N_CLASSES)
# Carrega o state_dict e mapeia para o CPU (caso o modelo tenha sido treinado em GPU)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval() # Coloca o modelo em modo de avaliação (importante!)

# --- 3. Definir Transformações da Imagem ---
# As transformações têm de ser as mesmas que foram usadas no treino.
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # Garante que a imagem é grayscale
    transforms.Resize((28, 28)), # Redimensiona para o tamanho que o modelo espera
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# --- 4. Inicializar a Aplicação Flask ---
app = Flask(__name__)

# --- 5. Definir as Rotas da Aplicação ---
@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    prediction = None
    if request.method == 'POST':
        # Verifica se um ficheiro foi enviado
        if 'image' not in request.files:
            return redirect(request.url)
        
        file = request.files['image']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Lê a imagem e processa-a
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            
            # Aplica as transformações
            image_tensor = transform(image).unsqueeze(0) # Adiciona uma dimensão de batch
            
            # Faz a previsão
            with torch.no_grad():
                output = model(image_tensor)
                _, predicted_idx = torch.max(output, 1)
                
            # Mapeia o resultado para o nome da classe
            prediction = CLASS_NAMES[predicted_idx.item()]
            
    # Renderiza a página HTML, passando a previsão se ela existir
    return render_template('index.html', prediction=prediction)

# --- 6. Correr a Aplicação ---
if __name__ == '__main__':
    app.run(debug=True)
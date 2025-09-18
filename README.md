# 🩺 Plataforma de Classificação de Cancro da Mama

## 📝 Descrição do Projeto

Este projeto é uma solução full-stack desenvolvida para um desafio de programação de IA, com o objetivo de criar uma ferramenta de apoio ao diagnóstico para o Sistema Nacional de Saúde Português. A aplicação é uma plataforma web que permite a um profissional de saúde fazer o upload de uma imagem histopatológica de tecido mamário e receber uma classificação imediata (Benigna ou Maligna) gerada por um modelo de Inteligência Artificial.

A solução foi construída de ponta a ponta, cobrindo o ciclo completo desde o treino do modelo de Deep Learning até à sua implementação numa interface web funcional.

---

## ✨ Funcionalidades

-   **Upload de Imagem:** Interface web simples para selecionar e enviar uma imagem.
-   **Análise por IA:** Utilização de uma Rede Neuronal Convolucional (CNN) para classificar a imagem em tempo real.
-   **Exibição de Diagnóstico:** O resultado da análise é apresentado de forma clara e imediata ao utilizador.

---

## 🛠️ Tecnologias Utilizadas

-   **Back-end:** Python, Flask
-   **Machine Learning:** PyTorch, MedMNIST
-   **Front-end:** HTML, CSS
-   **Bibliotecas de Suporte:** NumPy, Pillow (PIL)

---

## 🚀 Como Executar o Projeto Localmente

Para executar esta aplicação no seu próprio computador, siga os passos abaixo.

**Clonar o Repositório**

git clone [https://github.com/Aly-Filho/JuniorAI_avaliation_breastcancer.git](https://github.com/Aly-Filho/JuniorAI_avaliation_breastcancer.git)
cd JuniorAI_avaliation_breastcancer

**Criar e Ativar o Ambiente Virtual (venv)**
# Crie o ambiente virtual na pasta do projeto
python -m venv venv

# Ative-o (este comando é para o Command Prompt do Windows)
.\venv\Scripts\activate

# **Instalar as Dependências**
Com o ambiente ativo, instale todas as bibliotecas necessárias com um único comando:
pip install -r requirements.txt

# **Executar a Aplicação**
python app.py

# **Abrir no Navegador**
Visite http://127.0.0.1:5000 no seu navegador para usar a plataforma.


# 🛠️ Tecnologias Utilizadas
Back-end: Python, Flask

Machine Learning: PyTorch, MedMNIST

Front-end: HTML, CSS

Bibliotecas de Suporte: NumPy, Pillow

# Projeto de Classificação de Imagens com CIFAR-10 e Cálculo de Métricas de Avaliação

Este projeto demonstra a criação e treinamento de uma rede neural convolucional (CNN) para classificar imagens do conjunto de dados CIFAR-10 utilizando TensorFlow e Keras. Além disso, o código inclui a geração de uma matriz de confusão para análise de desempenho do modelo e um exemplo de cálculo das principais métricas de avaliação (sensibilidade, especificidade, acurácia, precisão e F-score) utilizando uma matriz de confusão arbitrária para um cenário de classificação binária.

## Visão Geral

- **Dataset:** CIFAR-10 (imagens coloridas com 10 classes)
- **Modelo:** Rede neural convolucional (CNN) com camadas de convolução, pooling, flatten e camadas densas.
- **Avaliação:** 
  - Geração de matriz de confusão com normalização e visualização via heatmap.
  - Cálculo das métricas de avaliação para um exemplo binário:
    - Sensibilidade (Recall): VP / (VP + FN)
    - Especificidade: VN / (FP + VN)
    - Acurácia: (VP + VN) / N (onde N = VP + FN + FP + VN)
    - Precisão: VP / (VP + FP)
    - F-score: 2 × (Precisão × Sensibilidade) / (Precisão + Sensibilidade)

## Requisitos

- Python 3.10.11
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/) (integrado ao TensorFlow)
- [Matplotlib](https://matplotlib.org/)
- [NumPy](https://numpy.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Pandas](https://pandas.pydata.org/)

## Instalação

1. Clone o repositório:

   ```bash
   git clone https://github.com/GabriD-0/Metricas_de_Validacao.git
   cd Metricas_de_Validacao
   ```

2. Instale as dependências:

   ```bash
   pip install tensorflow matplotlib numpy seaborn pandas
   ```

## Como Usar

1. Execute o script principal (por exemplo, `Main.py`):

   ```bash
   python Main.py
   ```

2. O script realizará as seguintes etapas:
   - **Download e Pré-processamento:** Carrega o dataset CIFAR-10 e normaliza as imagens.
   - **Treinamento:** Define e treina a CNN utilizando o conjunto de treinamento.
   - **Avaliação:** 
     - Realiza predições no conjunto de teste.
     - Gera e exibe uma matriz de confusão normalizada em um heatmap.
   - **Cálculo de Métricas:** Calcula e imprime as métricas de avaliação (sensibilidade, especificidade, acurácia, precisão e F-score) com base em uma matriz de confusão arbitrária para um exemplo de classificação binária.

## Estrutura do Código

- **Importações:** São utilizadas bibliotecas como TensorFlow, Keras, matplotlib, NumPy, Seaborn e Pandas.
- **Preparação dos Dados:** O dataset CIFAR-10 é carregado e as imagens são normalizadas.
- **Definição do Modelo:** A arquitetura da CNN é definida com camadas de convolução, pooling, flatten e densas. Observação: como as imagens são RGB, o `input_shape` é definido como `(32, 32, 3)`.
- **Treinamento e Validação:** O modelo é compilado com o otimizador `adam` e a função de perda `sparse_categorical_crossentropy`, e treinado por 2 épocas.
- **Matriz de Confusão e Visualização:** Após a predição, a matriz de confusão é calculada e normalizada, sendo visualizada com um heatmap.
- **Cálculo de Métricas de Avaliação:** Um exemplo de cálculo para classificação binária é apresentado, utilizando valores arbitrários de VP, FN, FP e VN.

## Métricas de Avaliação

Para a classificação binária, as seguintes métricas são calculadas:

- **Sensibilidade (Recall):**  
  \( \text{Sensibilidade} = \frac{VP}{VP + FN} \)

- **Especificidade:**  
  \( \text{Especificidade} = \frac{VN}{FP + VN} \)

- **Acurácia:**  
  \( \text{Acurácia} = \frac{VP + VN}{VP + FN + FP + VN} \)  
  *(Observação: \(N\) representa o número total de amostras.)*

- **Precisão:**  
  \( \text{Precisão} = \frac{VP}{VP + FP} \)

- **F-score:**  
  \( \text{F-score} = 2 \times \frac{\text{Precisão} \times \text{Sensibilidade}}{\text{Precisão} + \text{Sensibilidade}} \)

## Contribuição

Contribuições e Melhorias são bem-vindas! Caso deseje melhorar o projeto, sinta-se à vontade para criar _pull requests_ ou abrir _issues_ para discutir ideias.

## Licença

Este projeto está licenciado sob a [Licença MIT](LICENSE).

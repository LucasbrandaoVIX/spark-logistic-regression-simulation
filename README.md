# Simulação de treinamento distribuído com conceitos do Apache Spark

Este projeto usa corridas reais Yellow Taxi de janeiro de 2025 para comparar regressão logística manual em um processo com treinamento por gradiente agregado em processos Python separados. Não usa Apache Spark. O paralelismo é local, real e demonstrativo; não há cluster ou tolerância a falhas.

## Dados

Baixe `yellow_tripdata_2025-01.parquet` na seção January 2025 do [portal oficial TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) e coloque-o em `data/`. O portal publica os arquivos mensais em Parquet. O arquivo não é incluído neste projeto.

O programa mantém corridas com pagamento por cartão (`payment_type=1`), remove timestamps/valores inválidos e durações, distâncias, tarifas e contagens de passageiros fora de faixas plausíveis. A amostra é reprodutível. O alvo é `tip_amount / fare_amount >= 0.20`. Entradas: distância, tarifa, passageiros, duração e hora de início. `tip_amount` só é usado para calcular o alvo e não entra nas features. O split é estratificado; o scaler é ajustado apenas no treino.

## Instalação e execução

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py
```

A execução padrão usa 300 mil corridas válidas, 30 épocas, taxa 0,1 e compara 1, 2, 4 e 8 processos. Para controlar o experimento:

```bash
python train.py --data data/yellow_tripdata_2025-01.parquet --sample-size 100000 --workers 2 4 --epochs 30
```

`--sample-size` aceita 100000, 300000, 500000, 1000000 ou `0` (todas as válidas disponíveis). Saídas são gravadas em `outputs/`: `tempo_por_workers.png`, `loss_por_epoca.png` e `resultados.json`. O tempo medido cobre o treinamento, incluindo o overhead do pool de processos, mas exclui leitura, pré-processamento e avaliação. O speedup é tempo sequencial dividido pelo tempo daquela configuração; valores abaixo de 1 significam que o paralelismo foi mais lento.

### Resultado da execução real

Executado com o arquivo oficial de janeiro de 2025, amostra reprodutível de 300.000 corridas válidas, 80/20 estratificado, 30 épocas e taxa 0,1. O teste teve 60.000 corridas. Os resultados salvos em `outputs/resultados.json` foram:

A execução ocorreu no host RTX 5090; o treinamento usa processos de CPU via `multiprocessing`, sem CUDA ou aceleração da GPU.

| Processos | Tempo de treino | Speedup | Acurácia | F1 |
|---:|---:|---:|---:|---:|
| 1 (sequencial) | 0,114 s | 1,00× | 0,7603 | 0,8613 |
| 2 | 0,315 s | 0,36× | 0,7603 | 0,8613 |
| 4 | 0,240 s | 0,47× | 0,7603 | 0,8613 |
| 8 | 0,221 s | 0,51× | 0,7603 | 0,8613 |

Os pesos e as losses por época passaram a comparação numérica entre sequencial e distribuído. A loss caiu de 0,6931 para 0,5761. O melhor tempo distribuído ainda foi mais lento que o sequencial nesta amostra; o custo de criar tarefas e agregar resultados supera o trabalho pequeno de cada partição. Os tempos são de uma execução e podem variar conforme a carga da máquina.

## Arquitetura e relação com Spark

- **Driver:** processo principal que conserva os pesos, coordena as épocas e aplica a atualização do gradiente.
- **Partições:** fatias disjuntas do treino, uma por worker (ou menos se houver menos linhas que workers).
- **Workers:** processos criados por `ProcessPoolExecutor`; cada um calcula a soma do gradiente sobre sua partição.
- **Map:** calcular contribuições locais `(n_i, soma_gradiente_i, soma_loss_i)` para os pesos recebidos.
- **Reduce:** o driver soma as contribuições e divide por `N = soma(n_i)`. Isso produz gradiente e loss médios globais sem dar peso igual a partições de tamanhos diferentes.
- **Broadcast simulado:** os pesos atuais são enviados com cada tarefa a cada época. O driver atualiza `w = w - taxa * gradiente_global`.

O algoritmo sequencial usa o mesmo gradiente de batch, pesos iniciais zero e hiperparâmetros. O programa valida os pesos e a curva de loss de cada configuração com tolerância numérica antes de salvar os resultados. Em um cluster Spark real, dados e tarefas podem estar distribuídos entre máquinas; esta versão só demonstra o fluxo Map/Reduce por processos na mesma máquina.

## Roteiro breve para apresentação

1. **Spark:** mecanismo para processar grandes volumes de dados em paralelo em um cluster; aqui seus conceitos são ilustrados sem instalar Spark.
2. **Distribuição:** divide o conjunto em partes para diferentes processos trabalharem ao mesmo tempo.
3. **Driver e workers:** o driver coordena as épocas e atualiza o modelo; os workers processam partições.
4. **Map e Reduce:** cada worker calcula gradientes locais; o driver os soma e divide pelo total de exemplos.
5. **Broadcast:** todos recebem os mesmos pesos da época antes de calcular sua contribuição.
6. **Demonstração:** compare parâmetros/loss equivalentes, métricas no teste e tempos para 1, 2, 4 e 8 processos. O overhead local pode reduzir ou eliminar o speedup; isso é um resultado válido.
7. **Aplicação real:** em dados de mobilidade, cada partição pode processar registros de viagens; gradientes agregados permitem treinar um modelo sobre volumes que não cabem em uma única máquina, desde que a infraestrutura distribua armazenamento e tarefas.

## Limites didáticos

A duração e hora de pickup são derivadas dos timestamps. O alvo usa a tarifa como denominador, portanto corridas com tarifa não positiva são excluídas. Esta é uma simulação simplificada: serialização e cópia local substituem comunicação distribuída, e não são representados scheduler, armazenamento distribuído ou recuperação de falhas.

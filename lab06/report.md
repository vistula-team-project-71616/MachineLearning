
# Lab 6

Karina Romanova 71616

**Dataset:** Fashion-MNIST  
**Model:** CNN  
**Device:** CPU

## 1. Goal

The goal was to study and improve the model training pipeline. The experiments included training configuration, profiling, data loading, batch size comparison, learning rate scheduling, gradient accumulation and checkpoint resuming.

## 2. Training Pipeline

The CNN model from Lab 3 was used for the experiments. The baseline model was trained for 5 epochs using Adam with a learning rate of 0.001.

- Best validation accuracy: 89.76%.
- Test accuracy: 89.29%.


## 3. Training Profiling

| Stage | Time (s) | Percentage |
|---|---:|---:|
| Data Loading | 12.84 | 49.49% |
| Forward Pass | 5.95 | 22.95% |
| Backward Pass | 6.76 | 26.06% |
| Optimizer Step | 0.39 | 1.50% |

## 4. DataLoader Workers Study

| num_workers | Throughput (samples/s) |
|---:|---:|
| 0 | 5435.81 |
| 1 | 3284.40 |
| 2 | 3267.73 |
| 4 | 1477.56 |
| 8 | 1139.42 |

## 5. Batch Size Study

| Batch Size | Throughput (samples/s) | Peak Memory (MB) | Test Accuracy |
|---:|---:|---:|---:|
| 32 | 2438.68 | 423.05 | 89.96% |
| 64 | 2404.72 | 429.95 | 89.34% |
| 128 | 2429.59 | 328.58 | 88.80% |
| 256 | 2306.29 | 363.98 | 88.15% |
| 512 | 2247.74 | 251.26 | 86.12% |


## 6. Optimizers and Learning Rate Schedulers

| Optimizer | Scheduler | Test Accuracy |
|---|---|---:|
| Adam | StepLR | 89.26% |
| Adam | CosineAnnealingLR | 89.19% |
| Adam | ReduceLROnPlateau | 89.64% |
| SGD | StepLR | 88.87% |
| SGD | CosineAnnealingLR | 89.02% |
| SGD | ReduceLROnPlateau | 89.26% |

## 7. Gradient Accumulation

| Method | Effective Batch Size | Test Accuracy | Training Time (s) |
|---|---:|---:|---:|
| Batch size 256 | 256 | 87.94% | 109.14 |
| Batch 64 + accumulation 4 | 256 | 87.92% | 105.13 |

## 8. Checkpoint and Resume

| Epoch | Training Loss | Test Accuracy |
|---:|---:|---:|
| 1 | 0.6181 | 84.43% |
| 2 | 0.3862 | 87.07% |
| 3 | 0.3362 | 87.42% |
| 4 | 0.3166 | 88.21% |
| 5 | 0.2955 | 88.22% |

## 9. Conclusions

1. The CNN training pipeline was successfully configured using YAML and MLflow.
2. Data loading was the main bottleneck in the profiling experiment.
3. Using more DataLoader workers did not improve performance on the tested CPU.
4. Batch size 32 achieved the best accuracy and throughput in the batch size experiment.
5. Adam with ReduceLROnPlateau achieved the best test accuracy among the tested optimizer and scheduler combinations.
6. Gradient accumulation produced results very close to using a real batch size of 256.
7. Checkpoint resuming worked correctly and allowed training to continue after interruption.

## 10. Individual Project Proposal (approved)

### Title: Lightweight Art Style Recognition

**Student:** Karina Romanova 71616  
**Group:** 2

### Problem

The project aims to develop a model that can automatically recognize artistic styles in digital artwork. Such a system can help users explore paintings, assist designers in selecting visual styles and support artists in learning about different art movements.

Traditional classification models can require significant computational resources. Therefore, the project will focus on creating a lightweight model suitable for efficient inference.

### Dataset

The planned datasets are:

- **ArtBench-10:** a dataset containing more than 60,000 images divided into 10 artistic styles.
- **WikiArt:** a collection of artworks with labels for artistic styles, genres and artists.

### Proposed Solution

A Convolutional Neural Network (CNN) will be developed to classify artwork by artistic style.

The model will be optimized using pruning or knowledge distillation to reduce its size and inference time.

### Success Metrics

The project will be evaluated using:

- Classification accuracy.
- Macro F1-score.
- Inference latency of no more than 50 ms.
- Model size of no more than 5 MB.

### Deployment Constraints

The final model should be lightweight and suitable for deployment in an environment with limited computational resources.

### Planned Optimization

Pruning or knowledge distillation will be investigated to reduce model complexity while maintaining classification performance.

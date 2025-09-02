# CIFAR-10 Classification with MLP and CNN

This project explores **image classification on the CIFAR-10 dataset** using both a **Multi-Layer Perceptron (MLP)** and a **Convolutional Neural Network (CNN)**. The experiments analyze model performance, training efficiency, and the impact of **Batch Normalization, Dropout, and hyperparameter tuning**.

---

## 📦 Environment

- **Python**: 3.9.6  
- **PyTorch**  
- **Hardware**: MacBook Pro (Apple M1 chip)  

---

## ⚙️ Hyperparameters

| Parameter       | Value  |
|-----------------|--------|
| batch_size      | 100 |
| num_epochs      | 50 |
| learning_rate   | 0.001 |
| drop_rate       | 0.5 |

---

## 🧠 Models

### CNN Model
```python
Model(
 (conv1): Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
 (bn1): BatchNorm2d()
 (relu1): ReLU()
 (dropout1): Dropout()
 (maxpool1): MaxPool2d(2)
 (conv2): Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
 (bn2): BatchNorm2d()
 (relu2): ReLU()
 (dropout2): Dropout()
 (maxpool2): MaxPool2d(2)
 (fc): Linear(8192, 10)
 (loss): CrossEntropyLoss()
)
````

**Results**:

* Training accuracy: **88.37%**
* Validation accuracy: **77.06%**
* Best validation accuracy: **77.49%**
* Test accuracy: **76.92%**

---

### MLP Model

```python
Model(
 (fc1): Linear(3072, 512)
 (bn1): BatchNorm1d(512)
 (relu): ReLU()
 (dropout): Dropout(p=0.5)
 (fc2): Linear(512, 10)
 (loss): CrossEntropyLoss()
)
```

**Results**:

* Training accuracy: **69.92%**
* Validation accuracy: **55.20%**
* Best validation accuracy: **55.82%**
* Test accuracy: **55.40%**

---

## 📊 Key Findings

### Batch Normalization

* **CNN**: Removing BatchNorm reduced validation accuracy from **77.49% → 73.10%** (↓4.39%).
* **MLP**: Removing BatchNorm reduced validation accuracy from **55.82% → 53.18%** (↓2.64%).

### Dropout

* **CNN**: Removing dropout reduced validation accuracy from **77.49% → 75.26%** (↓2.23%).
* **MLP**: Removing dropout reduced validation accuracy from **55.82% → 53.97%** (↓1.85%).

> Overall, BatchNorm provided a **bigger accuracy improvement** than dropout.

---

## ⚡ Hyperparameter Tuning

### Batch Size

| Batch Size | Validation Accuracy |
| ---------- | ------------------- |
| 10         | 75.58%              |
| 100        | **78.68%**          |
| 1000       | 67.12%              |
| 10000      | 50.83%              |

👉 Best results at **batch\_size = 100**. Too-large batch sizes significantly degraded performance.

### Dropout Rate

| Dropout Rate | Validation Accuracy |
| ------------ | ------------------- |
| 0.0          | 73.45%              |
| 0.2          | **76.39%**          |
| 0.4          | 72.93%              |
| 0.6          | 70.10%              |
| 0.8          | 51.74%              |
| 1.0          | 10.09%              |

👉 Moderate dropout (0.2) improved generalization, but high dropout prevented learning.

---

## 📝 Conclusions

1. **CNNs outperformed MLPs** on CIFAR-10, achieving **77% vs 55% accuracy**.
2. **Batch Normalization** had a larger positive impact than **dropout**.
3. Optimal **batch size = 100** and **dropout = 0.2** provided the best trade-off.
4. MLPs were significantly faster to train but had much lower accuracy due to limited capacity for image data.

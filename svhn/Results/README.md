# Results of Classification on SVHN
All models were trained on SVHN (training data + extra data) for 20 epochs with Adam optimizer using 1e-4 learning rate.<br />
Partial plasticity implies plasticity used only on fully-connected layers, while full plasticity implies plasticity on CNNs too.

### Classification Accuracy
| Layers | Plasticity | Accuracy |
| ------ | ---------- | -------- |
| 20     | No         | 96.54    |
| 20     | Partial    | 96.58    |
| 20     | Yes        | 96.71    |
| 56     | No         | 96.92    |
| 56     | Partial    | 96.57    |
| 56     | Yes        | 96.94    |

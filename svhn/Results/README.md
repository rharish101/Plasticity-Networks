# Results of Classification on SVHN
All models were trained on SVHN (training data + extra data) for 20 epochs with Adam optimizer using 1e-4 learning rate.<br />
Partial plasticity implies plasticity used only on fully-connected layers, while full plasticity implies plasticity on CNNs too.

## Classification Accuracy
| Filename                                                             | Layers | Plasticity | Accuracy |
| -------------------------------------------------------------------- | ------ | ---------- | -------- |
| [Resnet20Screen.png](Resnet20Screen.png)                             | 20     | No         | 96.54    |
| [PartialPlasticResnet20Screen.png](PartialPlasticResnet20Screen.png) | 20     | Partial    | 96.58    |
| [PlasticResnet20Screen.png](PlasticResnet20Screen.png)               | 20     | Yes        | 96.71    |
| [Resnet56Screen.png](Resnet56Screen.png)                             | 56     | No         | 96.92    |
| [PartialPlasticResnet56Screen.png](PartialPlasticResnet56Screen.png) | 56     | Partial    | 96.57    |
| [PlasticResnet56Screen.png](PlasticResnet56Screen.png)               | 56     | Yes        | 96.94    |

## Training Curves

### Standard Resnet
#### 56 Layers
Accuracy
![resnet56\_accuracy](Resnet56Acc.png)
Loss
![resnet56\_loss](Resnet56Loss.png)
#### 20 Layers
Accuracy
![resnet20\_accuracy](Resnet20Acc.png)
Loss
![resnet20\_loss](Resnet20Loss.png)

### Resnet with Partial Plasticity
#### 56 Layers
Accuracy
![partial\_plastic\_resnet56\_accuracy](PartialPlasticResnet56Acc.png)
Loss
![partial\_plastic\_resnet56\_loss](PartialPlasticResnet56Loss.png)
#### 20 Layers
Accuracy
![partial\_plastic\_resnet20\_accuracy](PartialPlasticResnet20Acc.png)
Loss
![partial\_plastic\_resnet20\_loss](PartialPlasticResnet20Loss.png)

### Resnet with Full Plasticity
#### 56 Layers
Accuracy
![plastic\_resnet56\_accuracy](PlasticResnet56Acc.png)
Loss
![plastic\_resnet56\_loss](PlasticResnet56Loss.png)
#### 20 Layers
Accuracy
![plastic\_resnet20\_accuracy](PlasticResnet20Acc.png)
Loss
![plastic\_resnet20\_loss](PlasticResnet20Loss.png)

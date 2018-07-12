# Results of Classification on Pascal VOC2012
Models are based on the Darknet53 model in the paper: [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
All models were trained on Pascal VOC2012 for 20 epochs with Adam optimizer using 1e-4 learning rate.<br />
Plasticity on the upper layers indicates that plasticity was applied on the top 43 convolution layers.<br />
Plasticity on the lower layers indicates that plasticity was applied on the bottom 10 layers, ie. the lower 9 convolution layers and the fully connected layer.<br />

## Classification Accuracy
When validation data is used, 30% of the training dataset is randomly selected for validation, with early stopping based on the validation loss.<br />
When validation data is not used, the model is trained on the entire training dataset without early stopping.
| Filename                                                           | Validation | Plasticity   | Accuracy |
| --------------------------------------------------------------     | ---------- | ------------ | -------- |
| [DarknetScreen.png](DarknetScreen.png)                             | Yes        | No           | 90.19    |
| [DarknetScreen-2.png](DarknetScreen-2.png)                         | No         | No           | 90.65    |
| [FullyPlasticDarknetScreen.png](FullyPlasticDarknetScreen.png)     | Yes        | All layers   | 90.18    |
| [FullyPlasticDarknetScreen-2.png](FullyPlasticDarknetScreen-2.png) | No         | All layers   | 90.29    |
| [UpperPlasticDarknetScreen.png](UpperPlasticDarknetScreen.png)     | Yes        | Upper layers | 90.51    |
| [UpperPlasticDarknetScreen-2.png](UpperPlasticDarknetScreen-2.png) | No         | Upper layers | 90.59    |
| [LowerPlasticDarknetScreen.png](LowerPlasticDarknetScreen.png)     | Yes        | Lower layers | 90.23    |
| [LowerPlasticDarknetScreen-2.png](LowerPlasticDarknetScreen-2.png) | No         | Lower layers | 90.24    |

## Training Curves
The training curves have been obtained through Tensorboard after smoothing with a linear filter (default in Tensorboard) with the default value of 0.6.<br />
The orange curves represent training accuracy/loss.<br />
The blue curves represent validation accuracy/loss.

### Standard Darknet53
Accuracy
![darknet\_accuracy](DarknetAcc.png)
Loss
![darknet\_loss](DarknetLoss.png)

### Fully Plastic Darknet53
Accuracy
![fully\_plastic\_darknet\_accuracy](FullyPlasticDarknetAcc.png)
Loss
![fully\_plastic\_darknet\_loss](FullyPlasticDarknetLoss.png)

### Darknet53 with Plasticity on Upper Layers
Accuracy
![upper\_plastic\_darknet\_accuracy](UpperPlasticDarknetAcc.png)
Loss
![upper\_plastic\_darknet\_loss](UpperPlasticDarknetLoss.png)

### Darknet53 with Plasticity on Lower Layers
Accuracy
![lower\_plastic\_darknet\_accuracy](LowerPlasticDarknetAcc.png)
Loss
![lower\_plastic\_darknet\_loss](LowerPlasticDarknetLoss.png)

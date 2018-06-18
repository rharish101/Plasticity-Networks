# Results of Transfer Learning from SVHN to MNIST
All models were trained initially on SVHN with the same hyperparameters.<br />
For transfer learning, they were trained on MNIST training data for 200 epochs.<br />
Note that for some configurations there are 2 runs.

### Resnet20 with Plasticity
The fully-connected layer and the last stack of the model, ie. the last 7 convolutional layers were kept trainable.<br />
The other convolutional layers were kept frozen.

| Filename                       | Learning Rate | Batch Size | Accuracy |
| ------------------------------ | ------------- | ---------- | -------- |
| PlasticResnet20Mnist.png       | 1e-5          | 32         | 89.34    |
| PlasticResnet20Mnist2.png      | 1e-4          | 32         | 95.00    |
| PlasticResnet20Mnist2b8.png    | 1e-4          | 8          | 95.11    |
| PlasticResnet20Mnist2b16.png   | 1e-4          | 16         | 94.34    |
| PlasticResnet20Mnist2b16-2.png | 1e-4          | 16         | 95.14    |
| PlasticResnet20Mnist2b64.png   | 1e-4          | 64         | 93.03    |
| PlasticResnet20Mnist3.png      | 5e-4          | 32         | 96.57    |
| PlasticResnet20Mnist3-2.png    | 5e-4          | 32         | 95.59    |

### Standard Resnet20
The fully-connected layer and the last stack of the model, ie. the last 7 convolutional layers were kept trainable.<br />
The other convolutional layers were kept frozen.<br />
The batch size used is 32.

| Filename             | Learning Rate | Accuracy |
| -------------------  | ------------- | -------- |
| Resnet20Mnist.png    | 1e-5          | 66.05    |
| Resnet20Mnist2-2.png | 1e-4          | 91.60    |
| Resnet20Mnist2.png   | 1e-4          | 85.55    |
| Resnet20Mnist3.png   | 5e-4          | 90.60    |

### Resnet56 with Plasticity
The fully-connected layer and the last 6 convolutional layers were kept trainable.<br />
The other convolutional layers were kept frozen.<br />
The batch size used is 32.

| Filename                   | Learning Rate | Accuracy |
| -------------------------- | ------------- | -------- |
| PlasticResnet56Mnist.png   | 1e-4          | 74.91    |
| PlasticResnet56Mnist-2.png | 1e-4          | 78.48    |
| PlasticResnet56Mnist2.png  | 5e-4          | 83.74    |

In the following tests, the last 8 convolutional layers were kept trainable.

| Filename                    | Learning Rate | Accuracy |
| --------------------------  | ------------- | -------- |
| PlasticResnet56Mnist3.png   | 5e-4          | 86.48    |
| PlasticResnet56Mnist3-2.png | 5e-4          | 88.95    |

### Standard Resnet56
The fully-connected layer and the last 6 convolutional layers were kept trainable.<br />
The other convolutional layers were kept frozen.<br />
The batch size used is 32.

| Filename             | Learning Rate | Accuracy |
| -------------------  | ------------- | -------- |
| Resnet56Mnist.png    | 1e-4          | 88.52    |
| Resnet56Mnist2.png   | 5e-4          | 81.02    |
| Resnet56Mnist2-2.png | 5e-4          | 86.49    |

In the following tests, the last 8 convolutional layers were kept trainable.

| Filename            | Learning Rate | Accuracy |
| ------------------- | ------------- | -------- |
| Resnet56Mnist3.png  | 5e-4          | 91.22    |

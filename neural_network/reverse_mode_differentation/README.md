
# Neural Network

## Reverse Mode Automatic Differentation
In a neural network reverse mode automatic differentation is a technique for computing the derivative of the loss function with respect to the model parameters.

For binary classification the following loss function is used:

```Python
-1/batch_size * (label * log(sigmoid(logit)) + (1 - label) * log(1 - sigmoid(logit)))
```

The example below is based on a model with one hidden layer:
```Python
logit = weight * input + bias
```

|depth|adjoint||
|-|-|-|
|0|1||
|1|-1/batch_size|-1/batch_size * (label * log(sigmoid(logit)) + (1 - label) * log(1 - sigmoid(logit)))| 
|1|1|label * log(sigmoid(logit)) + (1 - label) * log(1 - sigmoid(logit))| 
|2 Left|label|label * log(sigmoid(logit))| 
|2 Left|1/sigmoid(logit)|log(sigmoid(logit))|
|2 Left|sigmoid(logit) * (1-sigmoid(logit))|sigmoid(logit)|
|2 Left|input|weight * input + bias|
|2 Right|(1 - label) |(1 - label) * log(1 - sigmoid(logit))
|2 Right|1/(1 - sigmoid(logit))  | log(1 - sigmoid(logit))
|2 Right|-1 | 1 - sigmoid(logit)
|2 Right|sigmoid(logit) * (1-sigmoid(logit)) | sigmoid(logit)
|2 Right|input |weight * input + bias


Gradient with respect to weight (left)
```Python
gradient_left = 1 * -1/batch_size * 1 * label * 1/sigmoid(logit) * sigmoid(logit) * (1-sigmoid(logit)) * input

gradient_left = 1/batch_size * -label * (1-sigmoid(logit)) * input
```

Gradient with respect to weight (right):
```Python
gradient_right = 1 * -1/batch_size * 1 * (1-label) * 1/(1 - sigmoid(logit)) * -1 * sigmoid(logit) * (1-sigmoid(logit)) * input

gradient_right = 1/batch_size * (1-label) * sigmoid(logit) * input
```

Gradient with respect to weight:
```Python
gradient = 1/batch_size * -label * (1-sigmoid(logit)) * input + 
           1/batch_size * (1-label) * sigmoid(logit) * input

gradient = 1/batch_size * input * (-label * (1-sigmoid(logit)) + (1-label) * sigmoid(logit) )

gradient = 1/batch_size * input * (sigmoid(logit) - label)
```

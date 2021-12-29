# Cross Entropy Loss

## Upper Bound

In case only the labels are known and not the inputs, the prediction that yields the lowest cross entropy is

``` Python
ratio_y1 = num_examples_y1 / num_examples

P(Y=1|X) = ratio_y1

cross_entropy_loss = ratio_y1 * -log(ratio_y1) + 
                     (1-ratio_y1) * -log((1-ratio_y1))
```

Example:

| Y | P(Y=1\|X) |
| ----- | ----- |
| 1 | 0.75 |
| 0 | 0.00 |
| 1 | 0.75 |
| 1 | 0.75 |

For this example the best prediction is
``` Python
P(Y=1|X) = 3/4 = 0.75

cross_entropy_loss = 3/4*-log(0.75) + 1/4 * -log(0.25) = 0.562335
```

## Lower Bound
In case all labels are completely random, the lowest cross entropy loss is achieved by predicting 0.5 for every input.

In this case the lowest cross entropy loss is `-log(0.5)`.

Example:
| X | Y | P(Y=1\|X) |
| - | - | - |
| 0 | 0 | 0.5000 |
| 0 | 1 | 0.5000 |
| 1 | 0 | 0.5000 |
| 1 | 1 | 0.5000 |

```
cross_entropy_loss = -log(0.5) = 0.693157
```

In general the lowest cross entropy loss is achieved by predicting the true conditional probability `P(Y=1|X)` for every input.
| X | Y | P(Y=1\|X) |
| - | - | - |
| 1 | 1 | 0.6667 |
| 1 | 0 | 0.6667 |
| 1 | 1 | 0.6667 |
| 0 | 0 | 0.0000 |

```
cross_entropy_loss = -log(0.5) = 0.562335
```

In case there are no latent variables, the lowest achievable cross entropy loss is 0.
| X | Y | P(Y=1\|X) |
| - | - | - |
| 1 | 1 | 1.0000 |
| 1 | 1 | 1.0000 |
| 1 | 1 | 1.0000 |
| 0 | 0 | 0.0000 |

```
cross_entropy_loss = 0.000000
```
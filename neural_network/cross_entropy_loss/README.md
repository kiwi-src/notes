# Cross Entropy Loss

## Upper Bound

In case only the labels Y are known and not the inputs X, the prediction that yields the lowest cross entropy is

``` Python
ratio_y1 = num_examples_y1 / num_examples

prob_y1 = ratio_y1

cross_entropy_loss = ratio_y1 * -log(prob_y1) + 
                     (1-ratio_y1) * -log((1-prob_y1))
```

### Example
Train dataset:

| Y |
| - |
| 1 |
| 0 |
| 1 | 
| 1 |

For this example the prediction with the smallest cross entropy is
``` Python
prob_y1 = 0.75
```

Cross entropy loss on train dataset:
``` Python
cross_entropy_loss = 3/4 * -log(3/4) + 
                     1/4 * -log(1/4) 
                   = 0.562335
```

Test dataset:
| Y | 
| - |
| 1 |
| 0 |
| 1 |

Predicting the learned probability `P(Y=1)=0.75` for every example in the test dataset leads to the following cross entropy loss:
``` Python
cross_entropy_loss = 2/3 * -log(3/4) + 
                     1/3 * -log(1/4) 
                   = 0.653886
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

``` Python
cross_entropy_loss = -log(0.5) = 0.693157
```

In general the lowest cross entropy loss is achieved by predicting the true conditional probability `P(Y=1|X)` for every input.
| X | Y | P(Y=1\|X) |
| - | - | - |
| 1 | 1 | 0.6667 |
| 1 | 0 | 0.6667 |
| 1 | 1 | 0.6667 |
| 0 | 0 | 0.0000 |

``` Python
cross_entropy_loss = 2/4 * -log(2/3) +
                     1/4 * -log(1/3) +
                     1/4 * -log(1) 
                   = 0.477386
```

In case there are no latent variables, the lowest achievable cross entropy loss is 0.
| X | Y | P(Y=1\|X) |
| - | - | - |
| 1 | 1 | 1.0000 |
| 1 | 1 | 1.0000 |
| 1 | 1 | 1.0000 |
| 0 | 0 | 0.0000 |

``` Python
cross_entropy_loss = 0.000000
```
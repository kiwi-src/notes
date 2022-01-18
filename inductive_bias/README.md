# Inductive Bias


## Dataset
For the analysis of different machine learning models and their inductive bias, the Breast Cancer Wisconsin Dataset is used. A subset of two features has been chosen: `worst_concave_point` and `worst smoothness`.

The histograms below show the distribution of the feature values. The Naive Bayes model assumes that the feature are distributed according to a Gaussian distribution. The real distribution of the features doesn't follow a Gaussian distribution (blue line).

<p align="center">
<img src="images/histograms.png" width="75%">
</p>


## Models
The conditional probability distributions of different machine learning models are plotted below.
<table>
<tr>
    <td>Model</td>
    <td style="text-align: center; vertical-align: middle;">Neural Network</td>
    <td style="text-align: center; vertical-align: middle;">XGBoost</td>
    <td style="text-align: center; vertical-align: middle;">Naive Bayes</td>
</tr>
<tr>
    <td>Validation Loss</td>
    <td style="text-align: center; vertical-align: middle;">0.1726</td>
    <td style="text-align: center; vertical-align: middle;">0.2708</td>
    <td style="text-align: center; vertical-align: middle;">0.2444</td>
</tr>
<tr>
    <td></td>
    <td><img src="images/neural-network.png"></td>
    <td><img src="images/xgboost.png"></td>
    <td><img src="images/naive-bayes.png"></td>
</tr>
</table>


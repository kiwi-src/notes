import time
import xgboost as xgb
import numpy as np
import utils
from dataset import Dataset
from utils import plot_color_mesh

if __name__ == '__main__':
    model_type = xgb.XGBClassifier

    kwargs = {
        'n_estimators': 12,
        'max_depth': 12,
        'colsample_bytree': 1.0,
        'colsample_bynode': 1.0,
        'colsample_bylevel': 1.0,
        'missing': -1,
        'use_label_encoder': False,
        'base_score': 0.5,
        'tree_method': 'exact',
        'booster': 'gbtree',
        'nthread': 1,
    }

    if model_type == xgb.XGBClassifier:
        kwargs['objective'] = 'binary:logistic'

    model = model_type(**kwargs)
    dataset = Dataset()
    train_features, train_labels, test_features, test_labels = dataset.load(
        format='np', batch_size=None)

    model.fit(train_features, train_labels,
              eval_set=[[test_features, test_labels]],
              verbose=True)
    best_iteration = model.get_booster().best_iteration

    feature_0 = np.linspace(0.0, 0.291, 1000)
    feature_1 = np.linspace(0.07117, 0.2226, 1000)
    mesh = utils.create_mesh(feature_0, feature_1)
    probs = model.predict_proba(mesh)[:, 1]

    plot_color_mesh(feature_0,
                    feature_1,
                    probs,
                    x_name='worst concave points',
                    y_name='worst smoothness',
                    inputs=train_features,
                    labels=train_labels,
                    filename='inductive_bias/images/xgboost.png')

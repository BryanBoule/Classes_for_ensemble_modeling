import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from OptimizeAUC import OptimizeAUC

if __name__ == '__main__':
    X, y = make_classification(n_samples=10000, n_features=25)

    # split into two folds
    xfold1, xfold2, yfold1, yfold2 = model_selection.train_test_split(X,
                                                                      y,
                                                                      test_size=0.5,
                                                                      stratify=y,
                                                                      random_state=42)

    # create 3 models
    logreg = linear_model.LogisticRegression()
    rf = ensemble.RandomForestClassifier()
    xgbc = xgb.XGBClassifier()

    # fit all models on fold1
    logreg.fit(xfold1, yfold1)
    rf.fit(xfold1, yfold1)
    xgbc.fit(xfold1, yfold1)

    # predict on fold2
    pred_logreg = logreg.predict_proba(xfold2)[:, 1]
    pred_rf = rf.predict_proba(xfold2)[:, 1]
    pred_xgbc = xgbc.predict_proba(xfold2)[:, 1]

    # average of all predictions
    avg_preds = (pred_logreg + pred_rf + pred_xgbc) / 3

    # get 2d array of predictions
    fold2_preds = np.column_stack((
        pred_logreg,
        pred_rf,
        pred_xgbc,
        avg_preds
    ))

    # calculate and store individual auc values
    aucs_fold2 = []
    for i in range(fold2_preds.shape[1]):
        auc = metrics.roc_auc_score(yfold2, fold2_preds[:, i])
        aucs_fold2.append(auc)

    print(f'Fold-2: LR AUC = {aucs_fold2[0]}')
    print(f'Fold-2: RF AUC = {aucs_fold2[1]}')
    print(f'Fold-2: XGB AUC = {aucs_fold2[2]}')
    print(f'Fold-2: AVG Preds AUC = {aucs_fold2[3]}')

    # P2

    logreg = linear_model.LogisticRegression()
    rf = ensemble.RandomForestClassifier()
    xgbc = xgb.XGBClassifier(use_label_encoder=False)

    # fit all models on fold1
    logreg.fit(xfold2, yfold2)
    rf.fit(xfold2, yfold2)
    xgbc.fit(xfold2, yfold2)

    # predict on fold2
    pred_logreg = logreg.predict_proba(xfold1)[:, 1]
    pred_rf = rf.predict_proba(xfold1)[:, 1]
    pred_xgbc = xgbc.predict_proba(xfold1)[:, 1]

    # average of all predictions
    avg_preds = (pred_logreg + pred_rf + pred_xgbc) / 3

    # get 2d array of predictions
    fold1_preds = np.column_stack((
        pred_logreg,
        pred_rf,
        pred_xgbc,
        avg_preds
    ))

    # calculate and store individual auc values
    aucs_fold1 = []
    for i in range(fold1_preds.shape[1]):
        auc = metrics.roc_auc_score(yfold1, fold1_preds[:, i])
        aucs_fold1.append(auc)

    print(f'Fold-1: LR AUC = {aucs_fold1[0]}')
    print(f'Fold-1: RF AUC = {aucs_fold1[1]}')
    print(f'Fold-1: XGB AUC = {aucs_fold1[2]}')
    print(f'Fold-1: AVG Preds AUC = {aucs_fold1[3]}')

    # P3
    # find optimal weight using the optimizer
    opt = OptimizeAUC()

    # delete the avg column
    opt.fit(fold1_preds[:, :-1], yfold1)
    opt_preds_fold2 = opt.predict(fold2_preds[:, :-1])
    auc = metrics.roc_auc_score(yfold2, opt_preds_fold2)
    print(f'Optimized AUC, Fold-2 = {auc}')
    print(f'Coefficients = {opt.coef_}')

    # find optimal weight using the optimizer
    opt = OptimizeAUC()

    # delete the avg column
    opt.fit(fold2_preds[:, :-1], yfold2)
    opt_preds_fold1 = opt.predict(fold1_preds[:, :-1])
    auc = metrics.roc_auc_score(yfold1, opt_preds_fold1)
    print(f'Optimized AUC, Fold-1 = {auc}')
    print(f'Coefficients = {opt.coef_}')

import statsmodels.api as sm
import pandas as pd
from sklearn.cross_validation import KFold
import itertools

"""
The function obtain the model fitting results
feature_set: is the collection of input predictors used in the model
X: is the dataframe containing all predictors to be enumerated
y: is the response vector
@return: a Series of quantities related to the model for model evaluation and selection
"""
def modelFitting(y, X, feature_set):
    # Fit model on feature_set and calculate RSS
    XX = X[list(feature_set)].copy();
    XX = sm.add_constant(XX)

    # fit the regression model
    model = sm.OLS(y,XX.values, hasconst=True).fit()
    return model;


"""
The function obtain the results given a regression model feature set
feature_set: is the collection of input predictors used in the model
X: is the dataframe containing all predictors to be enumerated
y: is the response vector
@return: a Series of quantities related to the model for model evaluation and selection
"""
def processSubset(y, X, feature_set):
    # Fit model on feature_set and calculate RSS
    regr = modelFitting(y, X, feature_set);
    R2 = regr.rsquared;
    ar2 = regr.rsquared_adj;
    sse = regr.ssr;
    return {"model":feature_set, "SSE": sse, "R2":R2, "AR2": ar2, "AIC": regr.aic, "BIC": regr.bic, "Pnum": len(feature_set)}


"""
The function find the regression results for all predictor combinations with fixed size
k: is the number of predictors (excluding constant)
X: is the dataframe containing all predictors to be enumerated
y: is the response vector
@return: a dataframe containing the regression results of the evaluated models
"""
def getAll(k, y, X):
    results = []
    # evaluate all the combinations with k predictors
    for combo in itertools.combinations(X.columns, k):
        results.append(processSubset(y, X, combo))

    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results);
    models['Pnum'] = k;
    print("Processed ", models.shape[0], "models on", k, "predictors")
    # Return the best model, along with some other useful information about the model
    return models


"""
The function find the Mallow's Cp based on the full model and existing regression results
models: is the dataframe containing the regression results of different models
fullmodel: is the model containing all predictors to calculate the Cp statistic
@return: a dataframe of models with Cp statistics calculated
"""
def getMallowCp(models, fullmodel):
    nobs = fullmodel.nobs;
    sigma2 = fullmodel.mse_resid;
    models['Cp'] = models['SSE']/sigma2 + 2*(models['Pnum']+1) - nobs
    return models

"""
The function find the best models among all lists using the criterion specified
models: is the dataframe containing the regression results of different models
criterion: is the selection critierion, can take values "AIC", "BIC", "Cp", "AR2", "R2" (only for educational purpose)
k: is the number of predictors as the constraints, if None, all models are compared
@return: the best model satisfied the requirement
"""
def findBest(models, criterion='AIC', k=None):
    # the list of models with given predictor number
    if k is None:
        submodels = models;
    else:
        submodels = models.loc[models['Pnum']==k,];

    # Use the criterion to find the best one
    if (criterion == "AR2") |  (criterion == "R2"):
        bestm = submodels.loc[submodels[criterion].idxmax(0), ];
    else:
        bestm = submodels.loc[submodels[criterion].idxmin(0), ];
    # return the selected model
    return bestm;


"""
The function use forward selection to find the best model given criterion
models: is the dataframe containing the regression results of different models
X: is the dataframe containing all predictors to be enumerated
y: is the response vector
criterion: is the selection critierion, can take values "AIC", "BIC", "Cp", "AR2", "R2" (only for educational purpose)
fullmodel: is the full model to evaluate the Cp criterion
@return: the best model selected by the function
"""
def forward(y, X, criterion="AIC", fullmodel = None):
    remaining = set(X.columns)   
    selected = []  
    current_score, best_new_score = float("inf"), float("inf")
    
    while remaining: # and current_score == best_new_score:
        scores_with_candidates = []
        
        for candidate in remaining:
            scores_with_candidates.append(processSubset(y, X, selected+[candidate]))
                        
        models = pd.DataFrame(scores_with_candidates)

        # if full model is provided, calculate the Cp
        if fullmodel is not None:
            models = getMallowCp(models, fullmodel);
            
        best_model = findBest(models, criterion, k=None)
        best_new_score = best_model[criterion];

        if (criterion == "AR2") |  (criterion == "R2"):
            best_new_score = -best_new_score;
        
        if current_score > best_new_score:
            selected = best_model['model'];
            remaining = [p for p in X.columns if p not in selected]
            print(selected)
            current_score = best_new_score
        else :
            break;
            
    model = modelFitting(y, X, selected)
    return model

"""
The function use backward elimination to find the best model given criterion
models: is the dataframe containing the regression results of different models
X: is the dataframe containing all predictors to be enumerated
y: is the response vector
criterion: is the selection critierion, can take values "AIC", "BIC", "Cp", "AR2", "R2" (only for educational purpose)
fullmodel: is the full model to evaluate the Cp criterion
@return: the best model selected by the function
"""
def backward(y, X, criterion="AIC", fullmodel = None):
    remaining = set(X.columns)   
    removed = []  
    current_score, best_new_score = float("inf"), float("inf")
    
    while remaining: # and current_score == best_new_score:
        scores_with_candidates = []
        
        for combo in itertools.combinations(remaining, len(remaining)-1):
            scores_with_candidates.append(processSubset(y, X, combo))
                        
        models = pd.DataFrame(scores_with_candidates)
        # if full model is provided, calculate the Cp
        if fullmodel is not None:
            models = getMallowCp(models, fullmodel);
            
        best_model = findBest(models, criterion, k=None)
        best_new_score = best_model[criterion];

        if (criterion == "AR2") |  (criterion == "R2"):
            best_new_score = -best_new_score;
                
        if current_score > best_new_score:
            remaining = best_model['model'];
            removed = [p for p in X.columns if p not in remaining]
            print(removed)
            current_score = best_new_score
        else :
            break;
            
    model = modelFitting(y, X, remaining)
    return model



"""
The function compute the cross validation results 
X: is the dataframe containing all predictors to be included
y: is the response vector
kf: is the kfold generated by the function
@return: the cross validated MSE 
"""
def CrossValidation(y, X, kf):
    results = []
    X = sm.add_constant(X);
    # evaluate all accuracy based on the folds
    for train_index, test_index in kf:
        d_train, d_test = X.ix[train_index,:], X.ix[test_index,:]
        y_train, y_test = y[train_index], y[test_index]

        # fit the model and evaluate the prediction
        lmfit = sm.OLS(y_train, d_train).fit();
        pred = lmfit.predict(d_test)
        prederror = ((pred - y_test) ** 2).mean();
        results.append(prederror);
        
    # Wrap everything up in a nice dataframe
    return results;

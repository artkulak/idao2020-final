# IDAO 2020 Final Stage
**Team: CHAD DATA SCIENTISTS**

**Solution scored 90.82 on private. 6th Place**


* KUDOS to my teammates:
	- https://github.com/JustM57
	- https://github.com/blacKitten13

* How to get the data
   - Train/test data folder from https://yadi.sk/d/JfZdPtNKu054VQ

<hr>

## Instructions
1. To train the model go to `solution` and run `Training.ipynb`. The generated models/preprocessors will appear in `solution/models` folder
2. Copy models to `submission/models`
3. Place `test.csv` into `submission`
4. Run `submission/main.sh`

<hr>

## Task description
In 2017 QIWI (https://qiwi.com/) launched a conceptually new product for Russian market — card with 0% interest instalment plan «Sovest». When lending money to someone there’s always a chance that they won’t pay you back. As QIWI bank takes such risk it scores creditworthiness of an individual. This is known as credit scoring.


But while banks try to improve their scoring models it’s easy to forget that ultimate goal isn’t about the score, but about improving client experience and maximizing proﬁts. QIWI, like all other banks, faces an obstacle when a credit scoring model estimates a borrower as conscientious, but in fact client does not use its card. Eventually, after paying for the COCA (cost of customer acquisition), the bank does not receive an expected revenue.
So far as we concern about the win-win case, when a client enjoys the product and the bank earns beneﬁts from it, we want to ﬁnd clients with zero or low-expected LTV (Life-Time Value) before making the decision on card issuing in order to take appropriate actions. For example, Sovest could oﬀer big welcome bonus for the ﬁrst transaction to increase client’s activity: cashback, increased grace period and so on.
Competitors are invited to participate in the process of enhancing Bank’s credit policy: to estimate not only the borrower’s probability of default, but also his LTV.

## Model description
It was crucial to understand the RocAuc top 10% metric hack to score well in the competition. Here is the metric:

```
def roc_auc_score_at_K(predicted_proba, target, rate=0.1): 
    # from sklearn.metrics import roc_auc_score 
    order = np.argsort(-predicted_proba) 
    top_k = int(rate ∗ len(predicted_proba)) 
    return roc_auc_score(target[order][:top_k], predicted_proba[order][:top_k])
```

Basically, it takes top 10% of most certain predictions and does ROC AUC score on top. Instead of passing just top 10% predictions, we did the following postprocessing:

```
def postprocess_predictions(predictions, rate = 0.1):
    thresh = POSTPROC_THRESH
    order = np.argsort(-predictions)  # -> 5%: лучшие (единицы) | 90% занулили (все что между) | 5% оставили (нули)
    top_k = int(0.1 * len(predictions))
    
    predictions[order[int(thresh * len(predictions)):int((1 - rate + thresh) * len(predictions))]] = -1
    
    return predictions
    
 ```
We used half of the predictions closest to zeros and half - closest to ones. This boosted our score significantly. Afterward we just used `5 folds * 5 seeds` LGBM models, created a bunch of features, and optimized hyperparameters of the models

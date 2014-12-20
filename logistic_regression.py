import pandas as pd
import statsmodels.api as sm
import numpy as np

loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')
loansData.dropna()

# Create Response Column
loansData['Interest.Rate'] = map(lambda x: float(str(x)[:-1]), loansData['Interest.Rate'])
loansData['IR.Below.12'] = map(lambda x: True if x <= 12 else False, loansData['Interest.Rate'])

# Create Predictors matrix
loansData['FICO.Score'] = map(lambda x: int(str(x).split("-")[0]), loansData['FICO.Range'])
ind_vars = loansData[['FICO.Score', 'Amount.Requested']]
ind_vars = sm.add_constant(ind_vars)

logit = sm.Logit(loansData['IR.Below.12'], ind_vars)
result = logit.fit()
coefs = result.params

def logistic_function(coefs, preds):
    total = sum(coefs*preds)
    return np.exp(total) / (1 + np.exp(total))

# Make a prediction
preds = np.array([1, 750, 10000])
p = logistic_function(coefs=coefs, preds=preds)
print ("The probability of a loan at less than 12 pct. interest"
       " with a FICO score of {} and loan size of {} is {:.2%}").format(
                                                                preds[1],
                                                                preds[2],
                                                                p)

def pred(model, preds, cutoff):
    p = logistic_function(model.params, preds)
    if p >= cutoff:
        return True
    else:
        return False

# Output with pred function
if pred(model=result, preds=preds, cutoff=0.7):
    print "The loan will be approved!"
else:
    print "The loan will not be approved!"
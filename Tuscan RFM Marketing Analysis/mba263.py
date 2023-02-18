import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind,norm
import statsmodels.api as sm
from matplotlib.pyplot import scatter as scatter2
from matplotlib.pyplot import plot
from matplotlib.pyplot import hist as hist2
from scipy.stats import pearsonr
import numpy
import collections
from pandas.api.types import is_numeric_dtype
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor
import sys
import warnings
warnings.simplefilter("ignore")

class Mba263RandomForest(RandomForestRegressor):
  def scale_fit(self, y, x):
    self.scaler = StandardScaler()
    self.scaler.fit(x)
    return self.fit(self.scaler.transform(x), y)
  def predict(self, x):
    return super(RandomForestRegressor, self).predict(self.scaler.transform(x))

class Mba263OLS(sm.OLS):
  def predict(self, *args):
    if args[1] is None:
      return super(sm.OLS, self).predict(args[0])
    else:
      x = args[1]
      if x.shape[0]==1:
        x = x.transpose()
      return super(sm.OLS, self).predict(args[0],sm.add_constant(x))

class Mba263Logit(sm.Logit):
  def predict(self, *args):
    if args[1] is None:
      return super(sm.Logit, self).predict(args[0])
    else:
      x = args[1]
      if x.shape[0]==1:
        x = x.transpose()
      return super(sm.Logit, self).predict(args[0],sm.add_constant(x))
    

class NeuralNetworkClassifier(MLPClassifier):
  def scale_fit(self, y, x):
    self.scaler = StandardScaler()
    self.scaler.fit(x)
    return self.fit(self.scaler.transform(x), y)
  def predict(self, x):
    return self.predict_proba(self.scaler.transform(x))[:,1]

class Mba263MNLogit(sm.MNLogit):
  def __init__(self,*args,**kargs):
    self.scaler = StandardScaler()
    self.scaler.fit(args[1])
    super(sm.MNLogit,self).__init__(args[0],self.scaler.transform(args[1]),**kargs)
  def predict(self, *args):
    if args[1] is None:
      return super(sm.MNLogit, self).predict(args[0])
    else:
      return super(sm.MNLogit, self).predict(args[0],self.scaler.transform(args[1]))

#    if len(args)==0:
#      return super(sm.MNLogit, self).predict()
#    else:
#      return super(sm.MNLogit, self).predict(self.scaler.transform(args[0]))

def neural_network(y,x,alpha=1e-5,hidden_layer_sizes=(5,5)):
  clf = NeuralNetworkClassifier(solver='lbfgs', alpha=alpha, hidden_layer_sizes=hidden_layer_sizes, random_state=1)
  clf.scale_fit(y, x)
  return clf

def random_forest(y,x,trees = 100, leaf_nodes=15):
  rf = Mba263RandomForest(n_estimators = trees, max_features=5, max_leaf_nodes=leaf_nodes, random_state = 42)
  rf.scale_fit(y, x)
  return rf

def rfm_sq(a,n):
  b = ntile(a[a.columns[0]],n) 

  c = numpy.zeros(len(a))
  d = numpy.zeros(len(a))
  for i in range(0,10):
    c[b==i]=10*ntile(a[b==i][a.columns[1]],n)+i
    for j in range(0,10):
      d[c==j*10+i]=100*ntile(a[c==j*10+i][a.columns[2]],n)+j*10+i

  return d

def sample_frequency(data,a):
  return data.loc[numpy.repeat(data.index.values,data[a])]

def ntile(a,n):
  q = a.quantile(numpy.linspace(1/n,1,n))
  output = []
  for i in a:
    if numpy.isnan(i):
      k = numpy.nan
    else:
      k = 0
      for j in q:
        if i<=j:
          break
        k += 1

    output.append(k)

  return numpy.array(output)

def get_means(b,c):
  m = {}
  n = {}
  for i in range(0,len(b)):
    if c[i] not in m:
      m[c[i]] = float(b[i])
      n[c[i]] = 1
    else:
      m[c[i]] += float(b[i])
      n[c[i]] += 1
  for key, value in m.items():
    m[key] = value / n[key]

  return pd.DataFrame([m[c[i]] for i in range(0,len(b))])

#  return pd.DataFrame([m[a[c].loc(i)] for i in range(0,len(a))])

def chi2(a,b):
  crosstab = pd.crosstab(a,b)
  chi2, p, dof, expected = chi2_contingency(crosstab)
    
  return chi2, p

def ttest(a,b):
  return ttest_ind(a,b)

def ttest_dummy(a,b,c):
  if all(x in [0,1] for x in a[b]):
    return ttest_ind(a[a[b]==1][c],a[a[b]==0][c])
  else:
    raise Exception("%s is not a dummy variable" % (b,))

def gain(res,p,bins=10):
  total = res.sum()
  grades = bins-1-ntile(p,bins)
  counts = numpy.zeros(bins+1)
#  totals = numpy.zeros(bins)
  for i in range(0,len(p)):
    if not numpy.isnan(grades[i]) and not numpy.isnan(res.iloc[i]):
      counts[int(grades[i])+1] += res.iloc[i]
#      totals[grades.loc[i]] += 1
  return numpy.cumsum(counts)/total

def lift(res,p,bins=10):
  total = res.sum()
  grades = bins-1-ntile(p,bins)
  counts = numpy.zeros(bins)
  totals = numpy.zeros(bins)
  for i in range(0,len(p)):
    if not numpy.isnan(grades[i]) and not numpy.isnan(res.iloc[i]):
      counts[int(grades[i])] += res.iloc[i]
      totals[int(grades[i])] += 1
  lift = numpy.cumsum(counts)/numpy.cumsum(totals)
  return lift/lift[-1]*100


def regress(a,b):
  mod = Mba263OLS(a, sm.add_constant(b), missing='drop')
  return mod.fit()

def odds_ratios(result_logit):
  odds=numpy.exp(result_logit.params[1:])
  se=numpy.exp(result_logit.params[1:])*result_logit.bse[1:]
  z=abs(odds-1)/se
  pvals=numpy.round(norm.sf(z)*2*1000)/1000
  lconf=odds-1.94*se
  rconf=odds+1.94*se
  return pd.DataFrame({'Odds ratios': odds, 'std err': se, 'z': z, 'P>|z|': pvals, '[0.025': lconf, '0.975]': rconf},index=result_logit.params.keys()[1:])
 
def mlogit(a,b):
  mod = Mba263MNLogit(a, b, missing='drop')
  return mod.fit()

def logit_reg(a,b,alpha=0):
  mod = Mba263Logit(a, sm.add_constant(b), missing='drop')
  return mod.fit_regularized(alpha=alpha,trim_mode='size',maxiter=1000)

def logit(a,b):
  mod = Mba263Logit(a, sm.add_constant(b), missing='drop')
  return mod.fit()

def scatter(a,b,c):
  return scatter2(a[b],a[c])

def plot_regression(a,b,c):
  mod = sm.OLS(a[b], sm.add_constant(a[c]), missing='drop')
  res = mod.fit()
  x = numpy.linspace(a[c].min(),a[c].max(),100)
  return plot(x,res.params[0]+x*res.params[1])

def pwcorr(a):
  n = len(a.columns)
  o = numpy.zeros((n,n))
  oo = numpy.zeros((n,n))
  for i in range (0,n):
    for j in range(i,n):
      rho, p = pearsonr(a.iloc[:,i],a.iloc[:,j])
      o[i][j]=rho
      o[j][i]=o[i][j]
      oo[i][j]=p
      oo[j][i]=oo[i][j]      

  return pd.DataFrame(o,columns=a.columns,index=a.columns), pd.DataFrame(oo,columns=a.columns,index=a.columns)

def dummy(a,b,c):
  res = ((x==c)*1+(x!=c)*0 for x in a[b])
  d = a.copy()
  name = b + '_dummy'  
  d[name] = list(res)

  return d

def encode(a,b,c):
  res = []
  for item in a[b]:
    found = False
    for key in c:
      if item==key:
        res.append(c[key])
        found = True
        break
    if not found:
      res.append(None)

  d = a.copy()
  name = b + '_enc' 
  d[name] = res

  return d

def hist(a,bins=10):
  a = a.dropna()
  if is_numeric_dtype(a):
    return hist2(a,bins)
  else:
    return a.value_counts().plot(kind='bar',color='#1f77b4')

def tabulate(a):
  counts = collections.Counter(a)
  output = pd.DataFrame()

  return pd.concat([pd.DataFrame([[i,counts[i],counts[i]/len(a)]], columns=['Name', 'Count', 'Frequency']) for i in counts],ignore_index=True).sort_values(['Name'])


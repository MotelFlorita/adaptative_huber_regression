from ada_hub.huber_regressor import HuberRegressor
import numpy as np
import scipy.stats
from sklearn.linear_model import LassoCV
n = 500
d = 1000

def generate_data(noise, n, d):
    beta = np.array([5, -2, 0, 0, 3])
    beta = np.concatenate((beta, np.zeros((max(d - 5, 0),))))
    x = scipy.stats.multivariate_normal(np.zeros((d,)), np.identity(d)).rvs(size=n)
    epsilon = noise.rvs(size=n)
    y = x @ beta + epsilon
    return x, y, beta

noises = [
    (scipy.stats.norm(0, 1), "normal"),
    (scipy.stats.t(df=1.5), "t-distrib"),
    (scipy.stats.lognorm(s=4, scale=np.exp(0)), "log-norm"),
]

x, y, optimal_beta = generate_data(noises[1][0], n, d)

"""
For numerical studies and real data analysis, in the case where the actual order
of moments is unspecified, we presume the variance is finite and therefore choose
robustification and regularization parameters as follows:
"""

c_tau = 0.5 # cross-val between {.5, 1, 1.5}
c_lambda = 0.1 # cross-val between {.5, 1, 1.5}

y_hat = np.mean(y)
sigma_hat = np.sqrt(np.mean((y - y_hat)**2))
n_eff = n / np.log(d)  # for simplicity
t = np.log(n)  # for simplicity
tau = c_tau*sigma_hat*np.sqrt(n_eff/t)

# Too strong in low dimension
#lambda_reg = c_lambda*sigma_hat*np.sqrt(n_eff/t)
lambda_reg = c_lambda*sigma_hat*np.sqrt(t/n_eff)

#lambda_reg = 0.5

print('Tau={}\nLambda={}'.format(tau, lambda_reg))

regressor = HuberRegressor(
    tau=tau, lambda_reg=lambda_reg, verbose='INFO',fit_intercept=True)

lasso = LassoCV(cv=3)
lasso.fit(x, y)
#beta_0 = lasso.coef_
beta_0 = np.zeros(d)

regressor.fit1(
    x, y, beta_0=beta_0, 
    phi_0=0.1, convergence_threshold=1e-8)

y_pred = regressor.predict(x)

#print(np.mean((y-y_pred)**2))
print("LassoCV(cv=3) loss:", np.sum((lasso.coef_ - optimal_beta)**2))
print("huber_regressor loss:", np.sum((regressor.coef_ - optimal_beta)**2))


print(lasso.coef_[0:6],regressor.coef_[0:6], optimal_beta[0:6], sep="\n")

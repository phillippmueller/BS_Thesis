# In[ ]:

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.integrate as integrate
from statsmodels.stats.power import zt_ind_solve_power 
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import normal_power 

# In[ ]:

n1, n2 = 29, 37
N = n1+n2
x1, x2 = 8, 26
p1_hat, p2_hat = x1/n1, x2/n2
p_hat = (x1+x2)/(n1+n2)

se = (p_hat*(1-p_hat)*((1/n1)+(1/n2)))**0.5 # standard error

mean_null = 0 # H0: mean2-mean1=0
mean_sample = p2_hat-p1_hat # sample: mean2-mean1= 0.7-0.28=0.42

# In[ ]:

# reported effect size = 0.426
# measured effect size = mean_sample = 0.42684
mean_reported = 0.426
z_measured = (mean_sample - mean_null)/se
p_measured = 2*stats.norm.sf(x=z_measured) # assuming two-tailed test was conducted
# reported p=0.0013:
z_reported = stats.norm.isf(q = 0.0013/2) # assuming two-tailed test was conducted 
# z = (effect - 0 /se) <=> se = effect/z
se_reported = mean_reported/z_reported
p_reported = 2*stats.norm.sf(x=z_reported) # assuming two-tailed test was conducted

print(f'measured effect size = {mean_sample}')
print(f'reported effect size = {mean_reported}')
print(f'measured z = {z_measured}')
print(f'reported z = {z_reported}')
print(f'measured se = {se}')
print(f'reported se = {se_reported}')
print(f'measured p = {p_measured}')
print(f'reported p = {p_reported}')
# se_reported - se
p_reported/p_measured
p_reported-p_measured

# In[ ]:

### power analysis: reported effect and se ###

# In[ ]:

sd_reported = se_reported * np.sqrt(N)
sd_measured = se*np.sqrt(N)
d_reported = mean_reported/sd_reported
d_measured = mean_sample/sd_measured
c_reported = stats.norm.isf(q=p_reported) - d_reported*np.sqrt(N)
c_measured = stats.norm.isf(q=p_measured) - d_measured*np.sqrt(N)
d_reported, d_measured

# In[ ]:

x = np.linspace(-5,5,100)
curve = stats.norm.pdf(x=x, loc=0, scale=1)
sns.lineplot(x=x, y=curve)
plt.axvline(c_reported)
plt.axvline(c_measured, color='orange')

# In[ ]:

power1 = stats.norm.sf(c_reported)
power2 = stats.norm.sf(c_measured)
print(power1)
print(power2)
power1-power2

# In[ ]:
### power analysis: estimated effects and reported se ###
# In[ ]:

effects = [.2, .4, .6, .8]
d = [effects[0]/sd_reported, effects[1]/sd_reported, effects[2]/sd_reported, effects[3]/sd_reported]
c_base = stats.norm.isf(q=p_reported)
c = [c_base-d[0]*np.sqrt(N), c_base-d[1]*np.sqrt(N), c_base-d[2]*np.sqrt(N), c_base-d[3]*np.sqrt(N)]
d

# In[ ]:

sns.lineplot(x=x, y=curve)
plt.axvline(c[0])
plt.axvline(c[1])
plt.axvline(c[2])
plt.axvline(c[3])
power = [stats.norm.sf(c[0]),stats.norm.sf(c[1]),stats.norm.sf(c[2]),stats.norm.sf(c[3])]
power

# In[ ]:
### power analysis: estimated effects and estimated se ###
# In[ ]:

d_est = [0.2, 0.5, 0.8]
c_est = [c_base-d_est[0]*np.sqrt(N), c_base-d_est[1]*np.sqrt(N), c_base-d_est[2]*np.sqrt(N)]
sns.lineplot(x=x, y=curve)
plt.axvline(c_est[0])
plt.axvline(c_est[1])
plt.axvline(c_est[2])
power = [stats.norm.sf(c_est[0]),stats.norm.sf(c_est[1]),stats.norm.sf(c_est[2])]
power

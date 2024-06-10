# In[ ]:

import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.weightstats import ztest
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.stats.power import normal_power 

# In[ ]:

data = pd.read_csv(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\data\heart.csv')
data

# In[ ]:
# exploratory analysis 
# In[ ]:

data.shape

# In[ ]:

data.nunique()

# In[ ]:

data.isnull().sum()

# In[ ]:

sns.pairplot(data)

# In[ ]:
# correlations 
# In[ ]:

cor = data.corr()
sns.heatmap(cor, cmap='coolwarm')
cor

# In[ ]:

normal_data = data[['age','trestbps','chol','thalach']]
normal_cor = normal_data.corr(method='pearson')
with sns.axes_style('darkgrid'):
    sns.heatmap(normal_cor, annot=True, cmap='coolwarm')
plt.savefig(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\plots\chapter6\plot1.png')

# In[ ]:
# test for significance of correlations 
# In[ ]:

# assumption: normal distribution
# kolmogorov smirnov test for normality
stats.kstest(rvs=normal_data.age, cdf='norm')# significant
stats.kstest(rvs=normal_data.trestbps, cdf='norm')# significant
stats.kstest(rvs=normal_data.chol, cdf='norm')# significant
stats.kstest(rvs=normal_data.thalach, cdf='norm')# significant 
# shapiro-wikls test 
stats.shapiro(normal_data.age)# significant
stats.shapiro(normal_data.trestbps)# significant
stats.shapiro(normal_data.chol)# significant
stats.shapiro(normal_data.thalach)# significant
# omnibus test
stats.normaltest(normal_data.age)# significant 
# stats.normaltest(normal_data.trestbps)# significant
# stats.normaltest(normal_data.chol)# significant
# stats.normaltest(normal_data.thalach)# significant

# 'all three tests reject H0:normal distribution for all four variables!'

# In[ ]:

# H1: thalach and age are correated 
n = len(normal_data)
df = n-2
r1 = normal_cor.thalach.age
# calculate test: Skript zur Vorlesung Analyse multivariater Daten, p.53
# H0: cor = 0
t = (r1*np.sqrt(n-2))/np.sqrt(1-r1**2) # test statistic
x = np.linspace(-10,10,1000)
sampling_dist = stats.t.pdf(x=x, df=df) # distribution under the null hypothesis
sns.lineplot(x=x, y=sampling_dist)
plt.axvline(t)
p_two = stats.t.cdf(x=t,df=df)*2
p_large = stats.t.sf(x=t, df=df)
p_small = stats.t.cdf(t, df=df)
print(f'two-tailed (H1:cor!=0): test statistic = {t}, p-value = {p_two}') 
print(f'right-tailed (H1:cor>0): test statistic = {t}, p-value = {p_large}') 
print(f'left-tailed (H1:cor<0): test statistic = {t}, p-value = {p_small}') 

# In[ ]:

# H1: age and trestbps are correlated
r2 = normal_cor.age.trestbps
t = (r2*np.sqrt(n-2))/np.sqrt(1-r2**2) # test statistic
sns.lineplot(x=x, y=sampling_dist)
plt.axvline(t)
p_two = stats.t.sf(x=t, df=df)*2
p_large = stats.t.sf(x=t, df=df)
p_small = stats.t.cdf(t, df=df)
print(f'two-tailed (H1:cor!=0): test statistic = {t}, p-value = {p_two}') 
print(f'right-tailed (H1:cor>0): test statistic = {t}, p-value = {p_large}') 
print(f'left-tailed (H1:cor<0): test statistic = {t}, p-value = {p_small}') 

# In[ ]:

# H1: age and chol are correlated
r3 = normal_cor.age.chol
t = (r3*np.sqrt(n-2))/np.sqrt(1-r3**2) # test statistic
sns.lineplot(x=x, y=sampling_dist)
plt.axvline(t)
p_two = stats.t.sf(x=t, df=df)*2
p_large = stats.t.sf(x=t, df=df)
p_small = stats.t.cdf(t, df=df)
print(f'two-tailed (H1:cor!=0): test statistic = {t}, p-value = {p_two}') 
print(f'right-tailed (H1:cor>0): test statistic = {t}, p-value = {p_large}') 
print(f'left-tailed (H1:cor<0): test statistic = {t}, p-value = {p_small}') 

# In[ ]:

# H1: chol and trestbps are correlated 
r4 = normal_cor.chol.trestbps
t = (r4*np.sqrt(n-2))/np.sqrt(1-r4**2) # test statistic
sns.lineplot(x=x, y=sampling_dist)
plt.axvline(t)
p_two = stats.t.sf(x=t, df=df)*2
p_large = stats.t.sf(x=t, df=df)
p_small = stats.t.cdf(t, df=df)
print(f'two-tailed (H1:cor!=0): test statistic = {t}, p-value = {p_two}') 
print(f'right-tailed (H1:cor>0): test statistic = {t}, p-value = {p_large}') 
print(f'left-tailed (H1:cor<0): test statistic = {t}, p-value = {p_small}') 

# In[ ]:

# H1:thalach and trestbps are correlated 
r5 = normal_cor.thalach.trestbps
t = (r5*np.sqrt(n-2))/np.sqrt(1-r5**2) # test statistic
sns.lineplot(x=x, y=sampling_dist)
plt.axvline(t)
p_two = stats.t.cdf(x=t, df=df)*2
p_large = stats.t.sf(x=t, df=df)
p_small = stats.t.cdf(t, df=df)
print(f'two-tailed (H1:cor!=0): test statistic = {t}, p-value = {p_two}') 
print(f'right-tailed (H1:cor>0): test statistic = {t}, p-value = {p_large}') 
print(f'left-tailed (H1:cor<0): test statistic = {t}, p-value = {p_small}') 

# In[ ]:

# pearson r test 
stats.pearsonr(x=data.thalach, y=data.age)# significant
stats.pearsonr(x=data.trestbps, y=data.age)# significant
stats.pearsonr(x=data.chol, y=data.age)# significant
stats.pearsonr(x=data.chol, y=data.trestbps)# significant
# stats.pearsonr(x=data.thalach, y=data.trestbps)# insignificant
# 'test decissions similar to result sabove'

# In[ ]:

a =  5.628106676351095e-13/5.628106676351095e-13
b = 7.762269074809919e-07/7.762269074809851e-07
c = 0.00017862864341450013/0.00017862864341449124
d = 0.032082053610872296/0.032082053610871034
a,b,c,d

# In[ ]:

# spearman rank correlation test 
stats.spearmanr(a=data.thalach, b=data.age) # significant 
stats.spearmanr(a=data.trestbps, b=data.age) # signficant
stats.spearmanr(a=data.chol, b=data.age) # significant
stats.spearmanr(a=data.chol, b=data.trestbps) # significant
# stats.spearmanr(a=data.thalach, b=data.trestbps) # insignificant

# In[ ]:

a = 6.024320734620622e-13/5.628106676351095e-13 
b = 4.2617094650125134e-07/7.762269074809851e-07
c = 0.0006099143222853829/0.00017862864341449124 
d = 0.027608539162108658/0.032082053610871034 
a,b,c,d

# In[ ]:

# sort by data type
n = len(data)
print(n)
binary = data[['sex','fbs','exang','target']]
# sex: 1=male
# fbs(fasting blood sugar > 120mg/dl): 1=true
# exang (exercise induced angina): 1=yes 
# target (diagnosis of heart disease): 0 = yes
numer = data[['age','trestbps','chol','thalach','oldpeak']]
# trestbps: resting blood pressure on admission to hospital
# chol: choloestorol levels 
#restecg: resting electroradiographic results
# thalach: maximum heart rate acchieved 
# oldpeak: ST depression induced by exercise relative to rest
# ca: number of major vessels colored by flourosopy 
categ = data[['cp','restecg','slope','thal','ca']]
# cp(chest pain type): 0=typical, 1=atypical, 2=non-anginal, 3=asymptomatic
# restecg (resting electrocardiographic resutls): 0=normal, 1=ST-T wave abnormal, 2=left ventricular hypertrophy
# slope (pf peak exercise ST segment): 0=up, 1=flat, 2=down
# thal: 1=normal, 2=fixed defect, 3=reversable defect 

# In[ ]:
# boxplots of numeric variables 
# In[ ]:

fig, axs = plt.subplots(2,3)
axs[0,0].boxplot(x=data.age)
axs[0,1].boxplot(x=data.trestbps)
axs[0,2].boxplot(x=data.chol)
axs[1,0].boxplot(x=data.thalach)
axs[1,1].boxplot(x=data.oldpeak)
axs[1,2].boxplot(x=data.ca)
axs[0,0].set_title('age')
axs[0,1].set_title('resting bps')
axs[0,2].set_title('cholestorol levels')
axs[1,0].set_title('max heart rate')
axs[1,1].set_title('ST depression time?')
axs[1,2].set_title('# major vessels')
fig.tight_layout()

# In[ ]:
# confidence intervals of binary variables 
# In[ ]:

p = binary.mean()
se_binary = np.sqrt(p*(1-p)/n)
z_score = stats.norm.ppf(0.975)

sex_ci = [p.sex-z_score*se_binary.sex,p.sex+z_score*se_binary.sex]
fbs_ci = [p.fbs-z_score*se_binary.fbs, p.fbs+z_score*se_binary.fbs]
exang_ci = [p.exang-z_score*se_binary.exang, p.exang+z_score*se_binary.exang]
target_ci = [p.target-z_score*se_binary.target, p.target+z_score*se_binary.target]

# In[ ]:

# plot 95% confidence intervals (dark)
# plt mean values (light)
with sns.axes_style('darkgrid'):
    plt.scatter(x='% male', y=sex_ci[0], color='darkblue')
    plt.scatter(x='% male', y=sex_ci[1], color='darkblue')
    plt.scatter(x='% male', y=p.sex, color='dodgerblue')
    plt.scatter(x='% fbs>120', y=fbs_ci[0], color='darkblue')
    plt.scatter(x='% fbs>120', y=fbs_ci[1], color='darkblue')
    plt.scatter(x='% fbs>120', y=p.fbs, color='dodgerblue')
    plt.scatter(x='% exercise \ninduced angina', y=exang_ci[0], color='darkblue')
    plt.scatter(x='% exercise \ninduced angina', y=exang_ci[1], color='darkblue')
    plt.scatter(x='% exercise \ninduced angina', y=p.exang, color='dodgerblue')
    plt.scatter(x='% no disease', y=target_ci[0], color='darkblue')
    plt.scatter(x='% no disease', y=target_ci[1], color='darkblue')
    plt.scatter(x='% no disease', y=p.target, color='dodgerblue')
    plt.title('95% confidende interval of binary variables')

# In[ ]:
# value count of categorical variables  
# In[ ]:

categ.mode()
# cp (chest pain type): 0=typical anigna, 1=atypical anigna, 2=non-anginal pain, 3=asymptomatic
# restecg (resting electrocardiographic results): 0=normal, 1=ST-T wave abnormality, 2=left ventricular hypertrophy
# slpoe (of peak exercise ST segment): 0=up, 1=flat, 2=down
# thal: 1=normal, 2=fixed defect, 3=reversable defect

# In[ ]:

t_ang, a_ang, n_ang, asymp = 0,0,0,0
norm, anorm, ventr = 0,0,0
up, flat, down = 0,0,0
normal, fixed, revers = 0,0,0
fail_cp, fail_restecg, fail_slope, fail_thal = [],[],[],[]
for i in range(len(categ)):
    if categ.cp.iloc[i] == 0:
        t_ang += 1
    elif categ.cp.iloc[i] == 1:
        a_ang += 1
    elif categ.cp.iloc[i] == 2:
        n_ang += 1
    elif categ.cp.iloc[i] == 3:
        asymp += 1
    else:
        fail_cp.append(i)
    if categ.restecg.iloc[i] == 0:
        norm += 1
    elif categ.restecg.iloc[i] == 1:
        anorm += 1
    elif categ.restecg.iloc[i] == 2:
        ventr += 1
    else:
        fail_restecg.append(i)
    if categ.slope.iloc[i] == 0:
        up += 1
    elif categ.slope.iloc[i] == 1:
        flat += 1
    elif categ.slope.iloc[i] == 2:
        down += 1
    else:
        fail_slope.append(i)
    if categ.thal.iloc[i] == 1:
        normal += 1
    elif categ.thal.iloc[i] == 2:
        fixed += 1
    elif categ.thal.iloc[i] == 3:
        revers += 1
    else:
        fail_thal.append(i)
print(fail_cp)
print(fail_restecg)
print(fail_slope)
print(fail_thal)

# In[ ]:

plt.bar(x='typical angina', height=t_ang)
plt.bar(x='atypical angina', height=a_ang)
plt.bar(x='non-aginal pain', height=n_ang)
plt.bar(x='asymptotic', height=asymp)
print(t_ang/n)
print(asymp/n)

# In[ ]:

plt.bar(x='normal', height=norm)
plt.bar(x='ST-T wave abnormality', height=anorm)
plt.bar(x='ventricular hypertrophy', height=ventr)
print(norm/n)
print(anorm/n)
print(ventr/303)

# In[ ]:

plt.bar(x='up', height=up)
plt.bar(x='flat', height=flat)
plt.bar(x='down', height=down)
print(up/303)
print(flat/303)
print(down/303)

# In[ ]:

plt.bar(x='normal', height=normal)
plt.bar(x='fixed defect', height=fixed)
plt.bar(x='reversable defect', height=revers)
plt.bar(x='measurement error', height=len(fail_thal))
print(normal)
print(normal/301)
print(fixed/303)

# In[ ]:
# multiple tests of mean on binary variables
# In[ ]:

disease, no_disease = binary.query('target==0'), binary.query('target==1')

m, f = binary.query('sex==1'), binary.query('sex==0')
male, female = m.drop(columns=['sex']), f.drop(columns=['sex'])

y,n_= binary.query('exang==1'), binary.query('exang==0')
yes, no = y.drop(columns=['exang']), n_.drop(columns=['exang'])

fy, fn = binary.query('fbs==1'), binary.query('fbs==0')
fbs_yes, fbs_no = fy.drop(columns=['fbs']), fn.drop(columns=['fbs'])

count_dis, count_nod = disease.sum(), no_disease.sum()
count_male, count_fem = male.sum(), female.sum()
count_yes, count_no = yes.sum(), no.sum()
count_fbsy, count_fbsn = fbs_yes.sum(), fbs_no.sum()

p_male = p.drop(index=['sex'])
p_yes = p.drop(index=['exang'])
p_fbs = p.drop(index=['fbs'])

# In[ ]:

dis_basel, nod_basel, dis_nod = [],[],[]
yes_basel, no_basel, yes_no = [],[],[]
male_basel, fem_basel, male_fem = [],[],[]
fbsy_basel, fbsn_basel, fbs_yes_no = [],[],[]
design = ['smaller','two-sided','larger']
for i in range(3):
    for d in range(3):
        # search for mean differences in target
        dis_basel.append(proportions_ztest(nobs=len(disease), count=count_dis[i], value=p[i], prop_var=True,  alternative=design[d])[0])
        dis_basel.append(proportions_ztest(nobs=len(disease), count=count_dis[i], value=p[i], alternative=design[d])[1])
        nod_basel.append(proportions_ztest(nobs=len(no_disease), count=count_nod[i], value=p[i], alternative=design[d])[0])
        nod_basel.append(proportions_ztest(nobs=len(no_disease), count=count_nod[i], value=p[i], alternative=design[d])[1])
        dis_nod.append(proportions_ztest(nobs=len(disease), count=count_dis[i], value=(count_nod[i]/len(no_disease)), alternative=design[d])[0])
        dis_nod.append(proportions_ztest(nobs=len(disease), count=count_dis[i], value=(count_nod[i]/len(no_disease)), alternative=design[d])[1])
        #search for mean differences in exang(exercise induced angina)
        yes_basel.append(proportions_ztest(nobs=len(yes), count=count_yes[i], value=p_yes[i], prop_var=True,alternative=design[d])[0])
        yes_basel.append(proportions_ztest(nobs=len(yes), count=count_yes[i], value=p_yes[i], alternative=design[d])[1])
        no_basel.append(proportions_ztest(nobs=len(no), count=count_no[i], value=p_yes[i], alternative=design[d])[0])
        no_basel.append(proportions_ztest(nobs=len(no), count=count_no[i], value=p_yes[i], alternative=design[d])[1])
        yes_no.append(proportions_ztest(nobs=len(yes), count=count_yes[i], value=(count_no[i]/len(no)), alternative=design[d])[0])
        yes_no.append(proportions_ztest(nobs=len(yes), count=count_yes[i], value=(count_no[i]/len(no)), alternative=design[d])[1])
        # search for mean differences in gender 
        male_basel.append(proportions_ztest(nobs=len(male), count=count_male[i], value=p_male[i], alternative=design[d])[0])
        male_basel.append(proportions_ztest(nobs=len(male), count=count_male[i], value=p_male[i], alternative=design[d])[1])
        fem_basel.append(proportions_ztest(nobs=len(female), count=count_fem[i], value=p_male[i], alternative=design[d])[0])
        fem_basel.append(proportions_ztest(nobs=len(female), count=count_fem[i], value=p_male[i], alternative=design[d])[1])
        male_fem.append(proportions_ztest(nobs=len(male), count=count_male[i], value=(count_fem[i]/len(female)), alternative=design[d])[0])
        male_fem.append(proportions_ztest(nobs=len(male), count=count_male[i], value=(count_fem[i]/len(female)), alternative=design[d])[1])
        # search for mean differences in fbs (fasting blood sugar)
        fbsy_basel.append(proportions_ztest(nobs=len(fbs_yes), count=count_fbsy[i], value=p_fbs[i], alternative=design[d])[0])
        fbsy_basel.append(proportions_ztest(nobs=len(fbs_yes), count=count_fbsy[i], value=p_fbs[i], alternative=design[d])[1])
        fbsn_basel.append(proportions_ztest(nobs=len(fbs_no), count=count_fbsn[i], value=p_fbs[i], alternative=design[d])[0])
        fbsn_basel.append(proportions_ztest(nobs=len(fbs_no), count=count_fbsn[i], value=p_fbs[i], alternative=design[d])[1])
        fbs_yes_no.append(proportions_ztest(nobs=len(fbs_yes), count=count_fbsy[i], value=(count_fbsn[i]/len(fbs_no)), alternative=design[d])[0])
        fbs_yes_no.append(proportions_ztest(nobs=len(fbs_yes), count=count_fbsy[i], value=(count_fbsn[i]/len(fbs_no)), alternative=design[d])[1])

target_iterables = [['sex','fbs','exang'],['left-tailed', 'two-tailed', 'right-tailed'],['statistic', 'p-value']]
sex_iterables = [['fbs','exang','target'],['left-tailed', 'two-tailed', 'right-tailed'],['statistic', 'p-value']]
exang_interables = [['sex','fbs','target'],['left-tailed', 'two-tailed', 'right-tailed'],['statistic', 'p-value']]
fbs_iterables = [['sex','exang','target'],['left-tailed', 'two-tailed', 'right-tailed'],['statistic', 'p-value']]

target_index = pd.MultiIndex.from_product(target_iterables, names=['test', 'design', 'result'])
sex_index = pd.MultiIndex.from_product(sex_iterables, names=['test', 'design', 'result'])
exang_index = pd.MultiIndex.from_product(exang_interables, names=['test', 'design', 'result'])
fbs_index = pd.MultiIndex.from_product(fbs_iterables, names=['test','design','result'])

df_dis_basel, df_nod_basel, df_dis_nod = pd.DataFrame(dis_basel,index=target_index), pd.DataFrame(nod_basel,index=target_index), pd.DataFrame(dis_nod,index=target_index)
df_male_basel, df_fem_basel, df_male_fem = pd.DataFrame(male_basel, index=sex_index), pd.DataFrame(fem_basel, index=sex_index), pd.DataFrame(male_fem, index=sex_index)
df_yes_basel, df_no_basel, df_yes_no = pd.DataFrame(yes_basel, index=exang_index), pd.DataFrame(no_basel, index=exang_index), pd.DataFrame(yes_no, index=exang_index)
df_fbs_yes_basel, df_fbs_no_basel, df_fbs_yes_no = pd.DataFrame(fbsy_basel, index=fbs_index), pd.DataFrame(fbsn_basel, index=fbs_index), pd.DataFrame(fbs_yes_no, index=fbs_index)

# In[ ]:

# couble check p-value calculation in stats.models.proportion -> statsmodels.stats.weighstats
def p_value (statistic):
    p_small = stats.norm.cdf(x=statistic)
    p_two = stats.norm.sf(x=statistic)*2
    p_large = stats.norm.sf(x=statistic)
    print(f'p_small={p_small}')
    print(f'p_two={p_two}')
    print(f'p_large={p_large}')
    
statistic=[]

for i in range(3): # number of characteristics to test
    a = proportions_ztest(nobs=len(disease), count=count_dis[i], value=(count_nod[i]/len(no_disease)), alternative='two-sided')[0] 
    statistic.append(a)
    if np.abs(a) > 1.646:
        print(f'significant finding detected in iteration {i}, statistic={a}')
        p_value(a)
        if np.abs(a) < 1.96:
            print(f'vague hypothesis detected in iteration {i}')
# results are correct!

# In[ ]:
# flexible sample size 
# In[ ]:
# delete latest observation on selected non-significant z-test findings 
# In[ ]:

# sex in (data, exang)
sex, e_sex = data.sex.values.tolist(), yes.sex.values.tolist()
statistic, p_val = [],[]
for i in range(len(e_sex)-2):
    count = np.sum(e_sex)
    p = np.mean(sex)
    statistic.append(proportions_ztest(nobs=len(e_sex), count=count, value=p, alternative='larger')[0])
    p_val.append(proportions_ztest(nobs=len(e_sex), count=count, value=p, alternative='larger')[1])
    if p_val[i] <= 0.05:
        print(p_val[i])
        print(i)
        break
    del(e_sex[len(e_sex)-1])
    del(sex[len(sex)-1])

# In[ ]:

# exang in (sex, data)
exang, m_exang = data.exang.values.tolist(), male.exang.values.tolist()
statistic, p_val = [],[]
for i in range(len(m_exang)-2):
    count = np.sum(m_exang)
    p = np.mean(exang)
    statistic.append(proportions_ztest(nobs=len(m_exang), count=count, value=p, alternative='larger')[0])
    p_val.append(proportions_ztest(nobs=len(m_exang), count=count, value=p, alternative='larger')[1])
    if p_val[i] <= 0.05:
        print(statistic[i])
        print(p_val[i])
        print(i)    
        # compute power on alpha = 5% and measured effect size
        mean = count/(len(m_exang)) - p
        se = mean/p_val[i]
        n = len(m_exang)+len(exang)
        d = mean/se*np.sqrt(n)
        c = stats.norm.isf(q=0.05) - d*np.sqrt(n)
        power = stats.norm.sf(c)
        print(f'power={power}')
        break
    del(m_exang[len(m_exang)-1])
    del(exang[len(exang)-1])

# In[ ]:

# fbs in (data, disease)
fbs, d_fbs = data.fbs.values.tolist(), disease.fbs.values.tolist()
statistic, p_val = [],[]
for i in range(len(d_fbs)-2):
    count = np.sum(d_fbs)
    p = np.mean(fbs)
    statistic.append(proportions_ztest(nobs=len(d_fbs), count=count, value=p, alternative='larger')[0])
    p_val.append(proportions_ztest(nobs=len(d_fbs), count=count, value=p, alternative='larger')[1])
    if p_val[i] <= 0.05:
        print(p_val[i])
        print(i)
        break
    del(d_fbs[len(d_fbs)-1])
    del(fbs[len(fbs)-1])

# In[ ]:

# fbs in (disease, no disease)
n_fbs, d_fbs = no_disease.fbs.values.tolist(), disease.fbs.values.tolist()
statistic, p_val = [],[]
for i in range(len(d_fbs)-2):
    count = np.sum(d_fbs)
    p = np.mean(n_fbs)
    statistic.append(proportions_ztest(nobs=len(d_fbs), count=count, value=p, alternative='larger')[0])
    p_val.append(proportions_ztest(nobs=len(d_fbs), count=count, value=p, alternative='larger')[1])
    if p_val[i] <= 0.05:
        print(p_val[i])
        print(i)
        break
    del(d_fbs[len(d_fbs)-1])
    del(n_fbs[len(n_fbs)-1])

# In[ ]:

# fbs in (male, female)
m_fbs, f_fbs = male.fbs.values.tolist(), female.fbs.values.tolist()
statistic, p_val = [],[]
for i in range(len(d_fbs)-2):
    count = np.sum(m_fbs)
    p = np.mean(f_fbs)
    statistic.append(proportions_ztest(nobs=len(m_fbs), count=count, value=p, alternative='larger')[0])
    p_val.append(proportions_ztest(nobs=len(m_fbs), count=count, value=p, alternative='larger')[1])
    if p_val[i] <= 0.05:
        print(p_val[i])
        break
    del(m_fbs[len(m_fbs)-1])
    del(f_fbs[len(f_fbs)-1])

# In[ ]:

# fbs in (exang no exang)
e_fbs, n_fbs = yes.fbs.values.tolist(), no.fbs.values.tolist()
statistic, p_val = [],[]
for i in range(len(d_fbs)-2):
    count = np.sum(e_fbs)
    p = np.mean(n_fbs)
    statistic.append(proportions_ztest(nobs=len(e_fbs), count=count, value=p, alternative='larger')[0])
    p_val.append(proportions_ztest(nobs=len(e_fbs), count=count, value=p, alternative='larger')[1])
    if p_val[i] <= 0.05:
        print(p_val[i])
        print(i)
        break
    del(e_fbs[len(e_fbs)-1])
    del(n_fbs[len(n_fbs)-1])

# In[ ]:
# try again with deleting outliers 
# In[ ]:

# exang in (sex, data)
exang, m_exang = data.exang.values.tolist(), male.exang.values.tolist()
statistic, p_val = [],[]
for i in range(len(m_exang)-2):
    count = np.sum(m_exang)
    p = np.mean(exang)
    statistic.append(proportions_ztest(nobs=len(m_exang), count=count, value=p, alternative='larger')[0])
    p_val.append(proportions_ztest(nobs=len(m_exang), count=count, value=p, alternative='larger')[1])
    if p_val[i] <= 0.05:
        print(statistic[i])
        print(p_val[i])
        print(f'sample1 size={len(m_exang)}, sample2 size={len(exang)}')
        # compute power on alpha = 5% and measured effect size
        mean = count/(len(m_exang)) - p
        se = mean/p_val[i]
        n = len(m_exang)+len(exang)
        d = mean/se*np.sqrt(n)
        c = stats.norm.isf(q=0.05) - d*np.sqrt(n)
        power = stats.norm.sf(c)
        print(f'power={power}')
        break
    if m_exang[i] == 0:
        del(m_exang[i])    
    if exang[i] == 1:
        del(exang[i])

# In[ ]:

# sex in (data, exang)
sex, e_sex = data.sex.values.tolist(), yes.sex.values.tolist()
statistic, p_val = [],[]
for i in range(len(e_sex)-2):
    count = np.sum(e_sex)
    p = np.mean(sex)
    statistic.append(proportions_ztest(nobs=len(e_sex), count=count, value=p, alternative='larger')[0])
    p_val.append(proportions_ztest(nobs=len(e_sex), count=count, value=p, alternative='larger')[1])
    if p_val[i] <= 0.05:
        print(statistic[i])
        print(p_val[i])
        print(f'sample1 size={len(e_sex)}, sample2 size={len(sex)}')
        # compute power on alpha = 5% and measured effect size
        mean = count/(len(e_sex)) - p
        se = mean/p_val[i]
        n = len(e_sex)+len(sex)
        d = mean/se*np.sqrt(n)
        c = stats.norm.isf(q=0.05) - d*np.sqrt(n)
        power = stats.norm.sf(c)
        print(f'power={power}')
        break
    if e_sex[i] == 0:
        del(e_sex[i])    
    if sex[i] == 1:
        del(sex[i])

# In[ ]:

# fbs in (data, disease)
fbs, d_fbs = data.fbs.values.tolist(), disease.fbs.values.tolist()
statistic, p_val = [],[]
for i in range(len(d_fbs)-2):
    count = np.sum(d_fbs)
    p = np.mean(fbs)
    statistic.append(proportions_ztest(nobs=len(d_fbs), count=count, value=p, alternative='larger')[0])
    p_val.append(proportions_ztest(nobs=len(d_fbs), count=count, value=p, alternative='larger')[1])
    if p_val[i] <= 0.05:
        print(p_val[i])
        print(f'sample1 size={len(d_fbs)}, sample2 size={len(fbs)}')
        # compute power on alpha = 5% and measured effect size
        mean = count/(len(d_fbs)) - p
        se = mean/p_val[i]
        n = len(d_fbs)+len(fbs)
        d = mean/se*np.sqrt(n)
        c = stats.norm.isf(q=0.05) - d*np.sqrt(n)
        power = stats.norm.sf(c)
        print(f'power={power}')
        break
    if d_fbs[i] == 0:
        del(d_fbs[i])    
    if fbs[i] == 1:
        del(fbs[i])

# In[ ]:

# fbs in (disease, no disease)
n_fbs, d_fbs = no_disease.fbs.values.tolist(), disease.fbs.values.tolist()
statistic, p_val = [],[]
for i in range(len(d_fbs)-2):
    count = np.sum(d_fbs)
    p = np.mean(n_fbs)
    statistic.append(proportions_ztest(nobs=len(d_fbs), count=count, value=p, alternative='larger')[0])
    p_val.append(proportions_ztest(nobs=len(d_fbs), count=count, value=p, alternative='larger')[1])
    if p_val[i] <= 0.05:
        print(p_val[i])
        print(f'sample1 size={len(d_fbs)}, sample2 size={len(n_fbs)}')
        # compute power on alpha = 5% and measured effect size
        mean = count/(len(d_fbs)) - p
        se = mean/p_val[i]
        n = len(d_fbs)+len(n_fbs)
        d = mean/se*np.sqrt(n)
        c = stats.norm.isf(q=0.05) - d*np.sqrt(n)
        power = stats.norm.sf(c)
        print(f'power={power}')
        break
    if d_fbs[i] == 0:
        del(d_fbs[i])    
    if n_fbs[i] == 1:
        del(n_fbs[i])
print('\ntest for opposite effect direction: \n')
n_fbs, d_fbs = no_disease.fbs.values.tolist(), disease.fbs.values.tolist()
statistic, p_val = [],[]
for i in range(len(n_fbs)-2):
    count = np.sum(n_fbs)
    p = np.mean(d_fbs)
    statistic.append(proportions_ztest(nobs=len(n_fbs), count=count, value=p, alternative='larger')[0])
    p_val.append(proportions_ztest(nobs=len(n_fbs), count=count, value=p, alternative='larger')[1])
    if p_val[i] <= 0.05:
        print(p_val[i])
        print(f'sample1 size={len(n_fbs)}, sample2 size={len(d_fbs)}')
        # compute power on alpha = 5% and measured effect size
        mean = count/(len(n_fbs)) - p
        se = mean/p_val[i]
        n = len(d_fbs)+len(n_fbs)
        d = mean/se*np.sqrt(n)
        c = stats.norm.isf(q=0.05) - d*np.sqrt(n)
        power = stats.norm.sf(c)
        print(f'power={power}')
        break
    if d_fbs[i] == 1:
        del(d_fbs[i])    
    if n_fbs[i] == 0:
        del(n_fbs[i])

# In[ ]:

# fbs in (male, female)
m_fbs, f_fbs = data.fbs.values.tolist(), disease.fbs.values.tolist()
statistic, p_val = [],[]
for i in range(len(m_fbs)-2):
    count = np.sum(m_fbs)
    p = np.mean(f_fbs)
    statistic.append(proportions_ztest(nobs=len(m_fbs), count=count, value=p, alternative='larger')[0])
    p_val.append(proportions_ztest(nobs=len(m_fbs), count=count, value=p, alternative='larger')[1])
    if p_val[i] <= 0.05:
        print(p_val[i])
        print(f'sample1 size={len(m_fbs)}, sample2 size={len(f_fbs)}')
        # compute power on alpha = 5% and measured effect size
        mean = count/(len(m_fbs)) - p
        se = mean/p_val[i]
        n = len(m_fbs)+len(f_fbs)
        d = mean/se*np.sqrt(n)
        c = stats.norm.isf(q=0.05) - d*np.sqrt(n)
        power = stats.norm.sf(c)
        print(f'power={power}')
        break
    if m_fbs[i] == 0:
        del(m_fbs[i])    
    if f_fbs[i] == 1:
        del(f_fbs[i])
        
print('\ntest for opposite effect direction: \n')

m_fbs, f_fbs = data.fbs.values.tolist(), disease.fbs.values.tolist()
statistic, p_val = [],[]
for i in range(len(f_fbs)-2):
    count = np.sum(f_fbs)
    p = np.mean(m_fbs)
    statistic.append(proportions_ztest(nobs=len(f_fbs), count=count, value=p, alternative='larger')[0])
    p_val.append(proportions_ztest(nobs=len(f_fbs), count=count, value=p, alternative='larger')[1])
    if p_val[i] <= 0.05:
        print(p_val[i])
        print(f'sample1 size={len(f_fbs)}, sample2 size={len(m_fbs)}')
        # compute power on alpha = 5% and measured effect size
        mean = count/(len(f_fbs)) - p
        se = mean/p_val[i]
        n = len(f_fbs)+len(m_fbs)
        d = mean/se*np.sqrt(n)
        c = stats.norm.isf(q=0.05) - d*np.sqrt(n)
        power = stats.norm.sf(c)
        print(f'power={power}')
        break
    if m_fbs[i] == 1:
        del(m_fbs[i])    
    if f_fbs[i] == 0:
        del(f_fbs[i])

# In[ ]:
# flexible sample size on numerical data: multiple small studies 
# In[ ]:

sns.pairplot(numer)

# In[ ]:

# test for differences between disease and no disease 
# sample size = 46, ngroups = 3
disease = data.query('target == 0') 
nodisease = data.query('target == 1')
disease1, nodisease1 = disease.iloc[0:46], nodisease.iloc[0:55]
disease2, nodisease2 = disease.iloc[46:92], nodisease.iloc[55:110]
disease3, nodisease3 = disease.iloc[92:138], nodisease.iloc[110:165]

# In[ ]:
# age
# In[ ]:

sns.histplot(disease1.age, color='navy')
sns.histplot(nodisease1.age, color='lightseagreen')

# In[ ]:

sns.histplot(disease2.age, color='navy')
sns.histplot(nodisease2.age, color='lightseagreen')

# In[ ]:

stats.levene(disease.age, nodisease.age)# significant
stats.levene(disease1.age, nodisease1.age)# significant
stats.levene(disease2.age, nodisease2.age)# insignificant
# stats.levene(disease3.age, nodisease3.age)# significant 

# stats.mannwhitneyu(x=disease.age, y=nodisease.age, alternative='larger') #significant 
# stats.mannwhitneyu(x=disease1.age, y=nodisease1.age, alternative='largerd') #significant
stats.mannwhitneyu(x=disease2.age, y=nodisease2.age, alternative='greater') #significant
# stats.mannwhitneyu(x=disease3.age, y=nodisease3.age, alternative='greater') # insignificant in 'two-sided' (significant in 'greater')

# In[ ]:

# try to reach equal variance in both groups (sub-set 1)
print(disease1.age.std()) # increase std 
print(nodisease1.age.std()) # decrease std -> del outlier  

# In[ ]:

group1, group2 = disease1.age.values.tolist(), nodisease1.age.values.tolist()
std = []
for i in range(len(group1)):
    t = stats.levene(group1, group2)[0]
    p = stats.levene(group1, group2)[1]
    std.append(np.std(group2))
    if p > 0.05:
        group2.remove(np.max(group2)) # pushes group2 down, better for later test of mean difference
    else:
        print(f'statsitic = {t}, p = {p}')
        print(len(group2))
        break
    if len(group2) < 2:
        break
stats.mannwhitneyu(group1, group2, alternative='greater') # significant

# In[ ]:

# try to reach equal variance in both groups (sub-set 2)
print(disease2.age.std()) # decrease std -> del outlier  
print(nodisease2.age.std()) # increase std

# In[ ]:

group1, group2 = disease2.age.values.tolist(), nodisease2.age.values.tolist()
for i in range(len(group2)):
    t = stats.levene(group1, group2)[0]
    p = stats.levene(group1, group2)[1]
    if p > 0.05:
        group1.remove(np.min(group1)) # pushes group1 up, better for later test of mean difference
    else:
        print(f'statsitic = {t}, p = {p}')
        print(len(group1))
        break
    if len(group1) < 2:
        break
stats.mannwhitneyu(group1, group2, alternative='greater') # significant

# In[ ]:
# oldpeak differences in both group (ST-deppression from exang, relative to rest)
# In[ ]:
# assumption violated: normal distribution (see histogram above)
# -> non-parametric test for mean differences
# In[ ]:

# test assumption: equal variance in both groups  
stats.levene(disease1.oldpeak, nodisease1.oldpeak)# significant
stats.levene(disease2.oldpeak, nodisease2.oldpeak)# significant
stats.levene(disease3.oldpeak, nodisease3.oldpeak)# significant 
stats.levene(disease.oldpeak, nodisease.oldpeak)# significant

# In[ ]:

# -> assumption satisfied: equal variance
stats.mannwhitneyu(x=disease.oldpeak, y=nodisease.oldpeak, alternative='greater') #significant 
stats.mannwhitneyu(x=disease1.oldpeak, y=nodisease1.oldpeak, alternative='greater') #significant
stats.mannwhitneyu(x=disease2.oldpeak, y=nodisease2.oldpeak, alternative='greater') #significant
stats.mannwhitneyu(x=disease3.oldpeak, y=nodisease3.oldpeak, alternative='greater') #significant

# In[ ]:

sns.histplot(data.oldpeak)

# In[ ]:

with sns.axes_style('darkgrid'):
    g = sns.histplot(data=data, x='oldpeak', hue='target', palette=['orangered','lightseagreen'])
    g.set(ylabel='absolute frequency', xlabel='ST depression induced by exercise (oldpeak) in mm')
plt.savefig(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\plots\chapter6\plot2.png')

# In[ ]:

sns.histplot(nodisease1.oldpeak, color='orange')
sns.histplot(disease1.oldpeak, color='blue')
print(nodisease1.oldpeak.std())
print(disease1.oldpeak.std())

# In[ ]:

sns.histplot(nodisease2.oldpeak, color='orange')
sns.histplot(disease2.oldpeak, color='blue')
print(nodisease2.oldpeak.std())
print(disease2.oldpeak.std())

# In[ ]:

sns.histplot(nodisease3.oldpeak, color='orange')
sns.histplot(disease3.oldpeak, color='blue')
print(nodisease3.oldpeak.std())
print(disease3.oldpeak.std())

# In[ ]:

group1 = disease1.oldpeak.values.tolist()
group2 = disease2.oldpeak.values.tolist()
group3 = disease3.oldpeak.values.tolist()
print(len(group1))
print(len(group2))
print(len(group3))
for i in range(len(data)):
    p1 = stats.levene(group1, nodisease1.oldpeak)[1]
    p2 = stats.levene(group2, nodisease2.oldpeak)[1]
    p3 = stats.levene(group3, nodisease3.oldpeak)[1]
    if p1 > 0.05:
        print('group1 is insignificant')
        print(stats.levene(group1, nodisease1.oldpeak))
        print(stats.mannwhitneyu(x=group1, y=nodisease1.oldpeak))
        print(len(group1))
    if p2 > 0.05:
        print('group2 is insignificant')
        print(stats.levene(group2, nodisease2.oldpeak))
        print(stats.mannwhitneyu(x=group2, y=nodisease2.oldpeak))
        print(len(group2))
    if p3 > 0.05:
        print('group3 is insignificant')
        print(stats.levene(group3, nodisease3.oldpeak))
        print(stats.mannwhitneyu(x=group3, y=nodisease3.oldpeak))
        print(len(group3))
    if p1 > 0.05 and p2 > 0.05 and p3 > 0.05:
        print(f'{stats.levene(group1, nodisease1.oldpeak)}, on sample size={len(group1)}')
        print(f'{stats.levene(group2, nodisease2.oldpeak)}, on sample size={len(group2)}')
        print(f'{stats.levene(group3, nodisease3.oldpeak)}, on sample size={len(group3)}')
        break
    else:
        group1.remove(np.min(group1))
        group2.remove(np.min(group2))
        group3.remove(np.min(group3))

# In[ ]:
# examplary study results
# prove relationship between sex and chol
# (false) inference: women eat more fast food 
# In[ ]:

sns.relplot(data=data, x='sex', y='chol')

# In[ ]:

data_female = data.query('sex==0')
data_male = data.query('sex==1')
mean_m = data_male.chol.mean()
std_m = data_male.chol.std()
mean_f = data_female.chol.mean()
std_f = data_female.chol.std()
n_f = len(data_female)
n_m = len(data_male)
pooled_sd = np.sqrt(((n_f-1)*std_f**2+(n_m-1)*std_m**2)/(n_f+n_m-2))
std_f, std_m # equal_va = False -> Welch's test

# In[ ]:

print([len(data_female),len(data_male)])
print([mean_f, mean_m])
ttest_ind(x1=data_female.chol,x2=data_male.chol, usevar = 'unequal', alternative='larger')

# In[ ]:

# controlling for all numeric variables 
print([data_female.age.median(), data_male.age.median()])
print([data_female.age.std(),data_male.age.std()])
ttest_ind(x1=data_female.age, x2=data_male.age, alternative='two-sided')# insignificant

# In[ ]:

print([data_female.trestbps.mean(), data_male.trestbps.mean()])
print([data_female.trestbps.std(),data_male.trestbps.std()])
ttest_ind(x1=data_female.trestbps, x2=data_male.trestbps, alternative='two-sided')# insignificant

# In[ ]:

# 95% confidence interval
effect = mean_f-mean_m
se = pooled_sd/np.sqrt(n_f+n_m)
l_bound = stats.norm.ppf(loc=effect, scale=se, q=0.025)
u_bound = stats.norm.ppf(loc=effect, scale=se, q=0.975)
conf = 0.975-0.025
conf

# In[ ]:

print([l_bound, effect, u_bound])
print([effect-l_bound, effect+u_bound])

# In[ ]:
# prove relationship between exang and target
# control for age
# In[ ]:

treatment, control = data.query('exang==1'), data.query('exang==0')
mean_treatment = len(treatment.query('target==0'))
mean_control = len(control.query('target==0'))
mean_treatment, mean_control # >5

# In[ ]:

ztest(disease.age, nodisease.age)

# In[ ]:

# show relationship between angina and target 
prop1 = mean_treatment/len(treatment)
prop2 = mean_control/len(control)
result = proportions_ztest(nobs=len(treatment), count=mean_treatment, value=prop2) # significant 
result

# In[ ]:

# control for age
# step1: levene test for equal group variance
stats.levene(treatment.age, control.age) # significant

# In[ ]:

sns.histplot(treatment.age, color='navy')
sns.histplot(control.age, color='lightseagreen')
print(len(treatment))
print(len(control))
treatment.age.std(), control.age.std()

# In[ ]:

# delete min and max from control.age to lower std and keep mean stable 
control = data.query('exang==0') # assuring no data loss!
p = stats.levene(treatment.age, control.age)[1]
print(f'p before:{p}')
while p <= 0.05:
    control = control.drop(control.age.idxmax())
    control = control.drop(control.age.idxmin())
    p = stats.levene(treatment.age, control.age)[1]
    print(p)
print(f'p after deleting {204-len(control)} outliers: {p}')

# In[ ]:

# step2: U-test for mean difference in both groups 
stats.mannwhitneyu(treatment.age, control.age) # insignificant

# In[ ]:

# repeat main test on cropped data
print(len(control)) # still cropped 
mean_treatment = len(treatment.query('target==0'))
mean_control = len(control.query('target==0'))
mean_treatment, mean_control # >5

# In[ ]:

prop1 = mean_treatment/len(treatment)
prop2 = mean_control/len(control)
new_result = proportions_ztest(nobs=len(treatment), count=mean_treatment, value=prop2) # significant 
print(new_result)
result[1]-new_result[1]# no change in p-values 

# In[ ]:
'Regional outcomes of severe acute respiratory syndrome coronavirus 2 infection on hospitalized patients with haematological malignancy'
# In[ ]:

# check age for normality
age = [1,1,
       2,2,2,2,2,2,2,2,2,2,
       3,3,3,3,3,3,3,3,3,3,3,3,
       4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
       5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]
group = [1,1,1,1,1,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,1,1,1,1,1, # treatment=1, control=0
        1,1,1,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,
        1,1,1,1,1,1,0,0,0,0,0,0,0]
sars_data = pd.DataFrame(data=[age,group], index=['age', 'group']).transpose()
stats.shapiro(age)# significant
stats.normaltest(age)# insignificant
stats.kstest(rvs=age, cdf='norm')# significant 

# In[ ]:
# attempt to replicate the controlling for age
# In[ ]:
# H0: pFCR in treatment > pFCR in control
# treatment: immunoyuppressive or cytotoxic medication
# controlling for age: employed method is not stated in the study 
# In[ ]:

# non-parametric test 
# assumption: equal variance 
stats.levene(sars_data.query('group==1').age, sars_data.query('group==0').age) # significant
# mean difference in age between treatment and control
stats.mannwhitneyu(x=sars_data.query('group==1').age, y=sars_data.query('group==0').age) # insignificant 

# In[ ]:

# parametric test
# mean difference in age between tratement and control
stats.ttest_ind(a=sars_data.query('group==1').age, b=sars_data.query('group==0').age) # insignficant 
ztest(x1=sars_data.query('group==1').age, x2=sars_data.query('group==0').age) # insignificant

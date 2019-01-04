
# coding: utf-8
Возьмем как пример распределение Лапласса
# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


a = 1.0
b = 0.0
ld = stat.laplace(loc=b, scale=a)
x = np.linspace(-4, 4, 1000)
plt.plot(x, stat.laplace.pdf(x))


# In[3]:


sample = ld.rvs(size=1000)
plt.hist(sample, normed=True, bins=30)
plt.ylabel('number of samples')
plt.xlabel('$x$')


# In[4]:


sizes = [5, 10, 30, 45]
samples = []
for i in sizes:
    samples.append(stat.laplace.rvs(size=[1000, i]))


# In[5]:


means_of_sample = []
for sample in samples:
    means = []
    for i in sample:
        means.append(np.mean(i))
    means_of_sample.append(means)


# In[6]:


mean = b
dispersion = 2 / ((2 * stat.laplace.pdf(mean)) ** 2)
print("mean: " + str(mean) 
      + "\ndispersion: " + str(dispersion))


# In[7]:


norm_rv = stat.norm(mean, np.sqrt(dispersion / sizes[0]))
x = np.linspace(-4, 4, 1000)
pdf = norm_rv.pdf(x)
plt.plot(x, pdf)

plt.hist(means_of_sample[0], normed=True)
plt.ylabel('number of samples')
plt.xlabel('n = 5')


# In[11]:


norm_rv = stat.norm(mean, np.sqrt(dispersion / sizes[1]))
x = np.linspace(-4, 4, 1000)
pdf = norm_rv.pdf(x)
plt.plot(x, pdf)

plt.hist(means_of_sample[1], normed=True)
plt.ylabel('number of samples')
plt.xlabel('n = 30')


# In[12]:


norm_rv = stat.norm(mean, np.sqrt(dispersion / sizes[2]))
x = np.linspace(-4, 4, 1000)
pdf = norm_rv.pdf(x)
plt.plot(x, pdf)

plt.hist(means_of_sample[2], normed=True)
plt.ylabel('number of samples')
plt.xlabel('n = 50')


# In[13]:


norm_rv = stat.norm(mean, np.sqrt(dispersion / sizes[3]))
x = np.linspace(-4, 4, 1000)
pdf = norm_rv.pdf(x)
plt.plot(x, pdf)

plt.hist(means_of_sample[3], normed=True)
plt.ylabel('number of samples')
plt.xlabel('n = 100')

Таким образом, чем больше объем выборки, тем больше гистограмма похожа на нормальное распределение
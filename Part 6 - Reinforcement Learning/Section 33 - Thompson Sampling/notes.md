# Thompson Sampling

- Can be used to solve multi-armed bandit problem
- uses bayesian inference
  - Ad *i* rewards y from Bernoulli distribution $p(y|\theta) = \Beta(\theta_i)$
  - $\theta$ is unknown but we set uncertainty by assuming uniform distribution
  - Bayes Rule: approach $\theta$ by the posterior distribution

$p(\theta_i|y) = \frac{p(y|\theta_i)p(\theta_i)}{\int(y|\theta_i)p(\theta_i)d\theta_i} proproportional p(y|\theta_i)Xp(\theta_i)$

- we get $p(\theta_i|y) = \beta$(number of successes + 1, number of failures + 1)
- at each round n, we take a random draw $\theta_i(n)$ from this posterior distribution $p(\theta_i|y)$, for each ad i
- at each round n we select the ad *i* that has the highest $\theta_i(n)$.

## Steps

1. at each round n, consider 2 numbers for each *ad i*:
   - $N_i^1(n)$ - the number of times the ad i got reward 1 up to round n
   - $N_i^0(n)$ - the number of times the ad i got reward 0 up to round n
2. for each ad i, we take a random draw from the distribution: $\theta_i(n) = \beta(N_i^1(n) + 1, N_i^0(n) + 1$
3. we select the ad that has the highest $\theta_i(n)$


- creating a distribution of where the expected value might lie

# Example

```python
import random
N = 10000
d = 10
ads_selected = []

numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d

total_reward = 0

for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0,d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1) 
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad] 
    if reward == 1:
        numbers_of_rewards_1[ad] += 1
    else:
        numbers_of_rewards_0[ad] += 1
        
    total_reward = total_reward + reward

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selected')     
plt.xlabel('Ads')
plt.ylabel('Number ad selected')
plt.show()   
  
```
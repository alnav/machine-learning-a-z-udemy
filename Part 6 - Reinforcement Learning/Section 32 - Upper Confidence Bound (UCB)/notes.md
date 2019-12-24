# Reinforcement learning

## Multi-armed bandit problem

One-armed bandit is a slot-machine

Multi-armed bandit is a set of slot-machines:

- how to play to maximise returns?
- assumption: each machine has different distribution, unknown
- find which machine has best distribution

- non-optimal machines cause regret (difference between best outcome and worst)
- problem is if a machine is not explored enough, sub-optimal might appear as best machine
- goal is to find best one while exploring all of them, but not too for too long

For example, have a set of ads, which one is better? Can only know after thousands of people look at them. Can be done with multiple A/B tests, but would spend lots of time and money on it, because would explore all option to the same degree.

## Upper confidence bound

steps:
1. have d arms (ads displayed)
2. each round, choose one ad to display
3. each round n, ad i gives reward 1 if user clicked on the ad, reward 0 if didn't
4. goal is to maximise total reward over many rounds

**Algorithm**:

1. at each round n, consider two numbers for each ad *i*:
   - Ni(n): number of times ad *i* was selected up to round n
   - Ri(n): sum of rewards of ad i up to round n
2. from these two numbers we compute
   - average reward of ad *i* up to round *n*:
      - r(n) = $\frac{Ri(n)}{Ni(n)}$
   - the confidence interval $[ri(n) - \Delta i(n), ri(n) + \Delta i(n)]$ at round n with:
      - $\Delta i(n) = \sqrt{\frac{3 log(n)}{2 Ni(n)}}$
3. select the ad *i* that has the maximum $UCB ri(n) + \Delta i(n)$  

with time, confidence interval becomes smaller, and initial equal rate of returns between machines converges to real rate for each machine

# Example

Random selection algorithm

- not very good, not improving with each iteration

```python
import random
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward
```

Upper Confidence Bound algorithm

```python
# Upper confidence bound algorithm
import math
N = 10000
d = 10
ads_selected = []

numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0

for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0,d):
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400 # objective is to run first 10 ads in order
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1 # update number of times ad was selected
    reward = dataset.values[n, ad] # get the correct reward from dataset
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
    
# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selected')     
plt.xlabel('Ads')
plt.ylabel('Number ad selected')
plt.show()   
```
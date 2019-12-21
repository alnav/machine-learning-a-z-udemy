# Naive Bayes

Think of wrenches, made by 2 machines, goal is to pick up defective ones.

What is the probability of machine 2 producing a defective wrench? Need to use **Bayes Theorem**:

$P(A|B) = \frac{P(B|A) * P(A)}{P(B)}$

Suppose initial information:
- Production
  - machine 1: 30/hr
  - machine 2: 20/hr
- Defective
  - 1% of total wrenches
  - 50% coming from machine 1
  - 50% coming from machine 2

Question:
- probability of part coming from machine 2 is defective?

Total produced per hour = 50
- P(mach1) = 30/50 = 0.6 *probability of wrench coming from machine 1*
- P(mach2) = 20/50 = 0.4 *probability of wrench coming from machine 2*
- P(defect) = 1%

P(Mach1 | defect) = 50%
P(Mach2 | defect) = 50%

  *| means given. likelyhood of part coming from machine 1, given condition that part is defective*

Machine 2 seems to produce disproportionally more defective parts.

## **Question is: P(defect | mach2) = ?**

can be intended as:
- what is probability of this part that just came out of machine 2, to be defective?
- what is proportion of parts coming out of machine 2, that are defective?

$P(A|B) = \frac{P(B|A) * P(A)}{P(B)}$

$P(defect | mach2) = \frac{P(mach2 | defect) * P(defect)}{P(mach2)}$

$P(defect | mach2) = \frac{0.5 * 0.01}{0.4} = 0.0125 = 1.25$ 

*1.25% or 12.5 every 1000 parts are defective*

## Example:

1000 wrenches
- 400 came from mach2
- 1% have a defect = 10
- of them, 50% came from mach2 = 5
- % defective wrenches from machine 2 = 5/400 = 1.25%

- probability of being defective if coming from machine 1:
  - $P(defect | mach1) = \frac{0.5 * 0.01}{0.6} = 0.0083 = 0.83$
  - 0.83%

# Use in machine learning

Probability of certain category, given features
- P(category|features)

$P(A|B) = \frac{P(B|A) * P(A)}{P(B)}$

**STEP 1** 

For first category, in order of calculation:
1. P(A): prior probability 
   - (e.g. $\frac{cat1}{cat1+cat2}$)
2. P(B): marginal likelihood
    - make a circle around data point, things inside are similar to our data point
    - number of similar observation / total observations
3. P(B|A): likelihood
    - circle around data point
    - number of similar observations among category / total number in the category
4. P(A|B): posterior probability

**STEP 2**

Same, for second category

**STEP 3**

Which category has higher probability for certain features?

# Assumptions

- Naive because relies on assumptions
  - variables we are working with, must be independent
- marginal probability does not change, when comparing 2 probabilities, does not matter
  - one less calculation to perform
  - $P(A|B) = P(B|A) * P(A)$
  - only if comparing the two!

# Example

```python
from sklearn.nave_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_test, y_test)
```
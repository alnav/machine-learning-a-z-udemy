# Apriori

*People who bought, also bought ...*.

Analyses associations

## Algorithm

Has 3 parts:
1. **support**

support(M1) = # user watchlists containing movie M1 / total number of watchlists

2. **confidence**

confidence(M1 -> M2) = # user watchlists containing M1 and M2 / # user watchlists containing M1

3. **lift**

lift(M1 -> M2) = confidence(M1 -> M2) / support(M1)  

Steps:

1. set a minimum support and confidence
2. take all subset in transactions having higher support than minimum support
3. take all the rules of these subsets having higher confidence than minimum confidence
4. sort the rules by decreasing lift

can later change value of minimum support and confidence, until satisfied with rules created

# Example

```python
# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# min support, for a product purchased 3 times a day:
# 3*7(days in a week)/7500 (total transactions) = 0.003

# if confidence set too high, 2 products might be associated just because they are very often bought overall
# e.g. people buy loads of mineral water, and also eggs. They will be associated


# Visualising the results
results = list(rules)
```

# Eclat model

Simplified apriori

1. support, same as apriori
   - support(x) = # user lists containing set x / # user lists
   - x is at least 2 things
2. no confidence or lift

**Steps:**

1. set minimum support
2. take all subsets in transactions having higher support than minimum
3. sort subsets by decreasing support




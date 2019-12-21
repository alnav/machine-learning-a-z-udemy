# Evualuating classification models performance

## False positive and negatives

- any classification will produce false positive and negatives
  - false positive: type 1 error
  - false negative: type 2 error
  - type 1 less dangerous than 2
    - something is going to happen, but won't
    - false negative is nothing is going to happen, but then it will!

## Confusion matrix

- identifies TP, TN, FP, FN
- top left TP, bottom right TN
- accuracy rate = correct / total
  - $\fract{TP + TN}{total}$
- error rate = wrong / total
  - $\fract{FP + FN}{total}$

## Accuracy paradox

- don't base judgment just on accuracy rate
- can be because a simple model can have high accuracy rate but be too crude to be useful
  - example: incidence of category A is 99%, then predicting every case as A will have accuracy 99%!

## CAP curve - cumulative accuracy profile

- larger area under curve, if better model
- can compare different models, by looking at CAP curves
- how much additional gain compared to random scenario or other model
- only topped by perfect model, which is very steep line

**ROC** - receiving operating characteristic

- not the same things as CAP, can look similar

## Evaluate CAP curves

- get area under perfect model curve, on top of next curve
- get area under model curve, on top of random model
- divide model curve / perfect curve
  - closer to 1 is good

Visually:
- look at 50% on x line, where it crosses the model, what is the value at y axis?
  - x < 60%: rubbish
  - 60% < x < 70%: poor
  - 70% < x < 80%: good
  - 80% < x < 90%: very good
  - above 90%: too good to believe ?overfitting
  - 100% something wrong with the variables ?post facto



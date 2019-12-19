# Kernel SVM

- SVM separates points by finding decision boundary

What happens when we can't find a boundary? Imagine a chart with one color at the center, with other color all around it like a donut

- For data that is not linearly separable
- Can separate by going to higher dimensional space

## Higher-dimensional space

- by finding a function that transform data point into higher dimension, 
- line can separate 1-dimension points 
- hyperplane can separate 2-dimension points
- If projected back into 2-dimension, separator would be a curve

### *Mapping to higher dimensional space can be highly compute-intensive*

- Issue solved by kernel trick!

## **Kernel trick**

## Gaussian rbf kernel 

$K(\overrightarrow{x},\overrightarrow{l^i}) = e-\frac{||\overrightarrow{x}-\overrightarrow{l^i}||^2}{2\sigma^2}$

- x: data point
- l: landmark from where distance is measured from $||x-l||$
- far away from landmar, big distance $e^{-distance} = 0$
- closer to landmark, small distance .. $e^0 = 1$
- function of $\sigma$: area of non-zero values gets larger for larger sigmas
- there can be multiple landmarks if shapes more complex than a circle

## Other types of kernal functions

- Sigmoid kernel
  - $K(X,Y) = tanh(y \cdot X^TY + r)$
- Polynomial kernal
  - $K(X,Y) = (y \cdot X^TY + r)^d,y > 0$




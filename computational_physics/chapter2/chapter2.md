# Chapter 2: Iterative - fractal

## Fractal

### Definiton

self-similarity

For instance: 山脉，羊齿叶，海岸线，**Mandelbrot set**

### The construction of Mandelbrot set

1. Iterative: top effient.

```
def f(n):
    solution=0
    for i in range(n):
        solution=iterative(solution)
    return solution
```

2. recursion: exquisite for human, but always slow effcient.有点像倒过来（实际运算还是正过来）的数学归纳法。

```
def f(n):
# The condition to stop this recursion
    if n=0:
        solution=0
    else:
        solution=interative(f(n))
    return solution
```

### The dim-computation of fractal

* Input:

For a square with length of side: 1, its area: 1

For a square with length of side: 2, its area: 4

$$
2^{dim}=4\\
dim=log_2 4=2
$$

* operation

For instance: Sierpinski三角形

$$
2^{dim}=3\\
dim=log_2 3
$$

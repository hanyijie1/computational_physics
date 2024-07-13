# Error Analysis

## Definition

* Error

$$
Error=Z_{real}-Z_{measure}
$$

* Absolute error

$\quad\quad$ Because $Z_{real}$ always cound not be precisely known, so we can only give a approximate range of Error:

$$
|Error|\leq\varepsilon
$$

$\quad\quad\varepsilon$ is absolute error, so we usually express $Z_{real}$ as:

$$
E_{real}=Error+Z_{measure}\in Z_{measure}\pm\varepsilon
$$

* Relative error

$$
\varepsilon_r=\frac{\varepsilon}{|Z_{real}|}\approx\frac{\varepsilon}{|Z_{measure}|}
$$

## Instance

* Round off

$\quad\quad$ Evidently, round off can only lead to half of the top digit precision, for instance:

$\quad\quad\pi\approx3.14\quad\varepsilon=0.005\quad\varepsilon_r=\frac{\varepsilon}{|3.14|}$

* multuple varables

$$
E=f(x_m,y_m)-f(x_r,y_r)\approx\partial_x f(x_r,y_r)(x_m-x_r)+\partial_y f(x_r,y_r)(y_m-y_r)=\partial_x E(x)+\partial_yE(y)
$$

## The type of error

1. Model error
2. Measurement error
3. Truncation error

$\quad\quad$Which generates from the operation from approximations made in turning mathematical operations (like derivatives) into discrete operations.

4. Round off

$\quad\quad$Which generates from the limited digits of floating number.

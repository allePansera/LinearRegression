# ML LinearRegression Intercept Term

Linear regression model with hyp under intercept form.

Hypotesis function is defined as: $$h_0(x) = \sum_{i=0}^{d-1} x_i*\theta_i$$


Loss function is defined as 'Least Squares': $$L(\theta) = \sum_{i=0}^{n-1} (h_0(x_i) - y_i)^2 $$

Loss funtion with Lasso Regression: $$J(\theta) = 0.5 * \sum_{i=0}^{n-1} (y_i - (\sum_{j=0}^{d-1} (\theta_j * x_ij)) )^2 $$

Feature selection is implemented with Wrapper Methods with Forward Selection

Implemented method for convex, BGD and SGD parameter optimized research

Run training.py to test theta param research and watch for the graphical comparison between each algorithm's result



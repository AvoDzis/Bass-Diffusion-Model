# Bass-Diffusion-Model
Frank Bass created the Bass diffusion model, which outlines how new items are adopted as an interaction between consumers and potential users. It is the mathematical model that describes the "s-curve" of innovation adoption. Along with the Dirichlet model of repeat buying and brand choice, it has been recognized as one of the most prominent empirical generalizations in marketing. The model is frequently used in forecasting, particularly in product and technology forecasting. The fundamental Bass diffusion equation is a Riccati equation with constant coefficients.


This equation can then be used to predict new adopters over time. That is why it is so valuable to marketers since it helps them to properly forecast future sales. One can utilize comparisons to previously launched products with comparable profiles even for new ones. Customer acquisition for new technologies and goods is based on the Bass Diffusion Model.


## How to run the code
```
import BassModel

model = BassModel.Bass_LSE('data.csv')
## or
model = BassModel.Bass_PolReg('data.csv')

model.fit() # fitting the data
model.predict() # getting the prediction
model.plot() # plot pdf predicted pdf against the actual sales datapoints
```
## Different Classes Explanation
There are two different methods that are utilized for each of these regressions.

- **OLS:** this method tries to to estimate the best coefficients in order to the total sum of squares of the difference between the calculated and observed values of y, is minimised. Then we try to pass those paramters back to obtain the coefficient of imitation, innovation, and the maximum number of adopters

- **curv_fit:** this method uses non-linear least squares in order to estimate its coefficients. However, It's Levenberg-Marquadt nonlinear fitting for unbounded problems and a trust-region variant when bounds are given. 

## Data
This package expects the user to have sales data with two columns set in the following order. 

- **Date:** this could of couse be the daily/monthly/quarterly sales data

- **Sales:** the sales that was seen in that given timeframe

It goes without saying that the inserted data should reference either a file format that corresponds to CSV or XLSX

## Plots

This package aims to fit the given data and based on the algorithms explained above make a refression from it in order to further predict what the sales would look like. Then we try to make a plot based on the information generated

- **plot():** PDF visualization of the predicted sales and actual sales. It plots the scatterplot of actual sales and regression of predicted sales

- **plot_cdf():** CDF visualization of the predicted sales. It plots the regression of cumulative predicted sales

## References

Bass diffusion model: https://www.immagic.com/eLibrary/ARCHIVES/GENERAL/WIKIPEDI/W101203B.pdf [Accessed May 8th 2022]

Helpful GitHub repos:

By u/alejandropuerto https://github.com/alejandropuerto/product-market-forecasting-bass-model/blob/master/Bass%20Model.ipynb

by u/Fahad021 https://github.com/Fahad021/Bass-Difussion-Modell-with-python/blob/master/Bass%20diffusion%20model.ipynb

By u/NForouzandehmehr https://github.com/NForouzandehmehr/Bass-Diffusion-model-for-short-life-cycle-products-sales-prediction/blob/master/bass.py

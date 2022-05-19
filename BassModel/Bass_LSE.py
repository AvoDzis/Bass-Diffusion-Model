import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import pandas as pd
from Bass import Bass


class Bass_LSE(Bass):
    """
    OLS this method tries to estimate the best coefficients in order to the total sum of squares of the difference between the calculated and observed values of y, is minimised. Then we try to pass those paramters back to obtain the coefficient of imitation, innovation, and the maximum number of adopters
    """
    def __init__(self, filename=None):
        """
         Initializing our class to get the Bass Model
         Parameters
         ----------
         filename : pandas.DataFrame
             Data to fit the model.
         Returns
         -------
         None
         """
        if filename is None:
            print('Error: must input filename.')
        try:
            if filename[-3:] == 'csv':
                delimiter = ','
            else:
                delimiter = '\t'
            self.data = np.genfromtxt(filename, delimiter=delimiter)
        except OSError:
            print('Error: file not found. Please input valid filename.')

        try:
            self.response = self.data[:, 1]
            self.time = range(1, len(self.response) + 1)
            self.title = filename[: 2]

        except IndexError:

            print('Error: incorrect file format. Please provide either csv or txt.')

        self.sales = None
        self.sales_forecast = None
        self.cumsales = None
        self.cumsales_forecast=None
        self.mod = None
        self.cum_sales_squared = None
        self.res = None
        self.pars = None
        self.m = None
        self.p = None
        self.q = None

    def fit(self):
        """
         Method to fit the data to the Bass Model OLS regression
         Returns
         -------
         float
            Obtain the coeficient estimation from the OLS regression
         """
        self.sales = self.response
        self.cumsales = np.cumsum(self.sales)
        self.cum_sales_squared = self.cumsales ** 2 #making cumulative sales column for later plots
        list_of_tuples = list(zip(self.sales, self.cumsales, self.cum_sales_squared))
        df = pd.DataFrame(list_of_tuples,
                          columns=['sales', 'cumsales', 'cum_sales_squared'])
        self.mod = smf.ols(formula='sales ~ cumsales + cum_sales_squared', data=df)
        self.res = self.mod.fit()
        self.pars = self.res.params
        return self.pars

    def __bass_model__(self, p, q, t):
        """
        Obtain rate of change of installed base
        Parameters
        ----------
        p : int
            coefficient of innovation
        p : int
            coefficient of imitation
        t: int
        Returns
        -------
        Obtain rate of change of installed base
        """
        return (np.exp((p + q) * t) * p * (p + q) ** 2) / ((p * np.exp((p + q) * t) + q) ** 2)

    def __max__(self, a, b):
        """
        Dunder method for max
        Parameters
        ----------
        a : int

        b : int
        -------
        max value between a and b
        """
        return max(a, b)

    def predict(self):
        """
         Getting the coefficient of imitation, innovation, and the maximum number of adopters
         Returns
         -------
         float
            coefficient of imitation, innovation, and the maximum number of adopters
         """
        m1 = (-self.pars['cumsales'] + np.sqrt(
            self.pars['cumsales'] ** 2 - 4 * self.pars['Intercept'] * self.pars['cum_sales_squared'])) / (
                     2 * self.pars['cum_sales_squared'])
        m2 = (-self.pars['cumsales'] - np.sqrt(
            self.pars['cumsales'] ** 2 - 4 * self.pars['Intercept'] * self.pars['cum_sales_squared'])) / (
                     2 * self.pars['cum_sales_squared'])
        self.m = self.__max__(m1, m2) #max potential
        self.p = self.pars['Intercept'] / self.m #innovation
        self.q = -self.m * self.pars['cum_sales_squared'] #imitation
        # print(self.m, self.p, self.q)
        self.sales_forecast = self.m * self.__bass_model__(self.p, self.q, self.time)
        return self.m, self.p, self.q

    def plot(self):
        """
        PDF visualization of the predicted sales and actual sales
        Returns
        -------
        scatterplot of actual sales and regression of predicted sales
        """

        plt.plot(self.time, self.sales, 'o', color='black', label='Actual Sales')
        plt.plot(self.time, self.sales_forecast, label='Sales Forecast')
        plt.ylabel('Sales')
        plt.xlabel('Time')
        plt.title("Bass Model for Product Sales")
        plt.legend()
        plt.show()

    def plot_cdf(self):
        """
        CDF visualization of the predicted sales
        Returns
        -------
        Regression of cumulative predicted sales
        """
        self.cumsales_forecast = np.cumsum(self.sales_forecast)
        plt.plot(self.time, self.cumsales_forecast, label='CProb', color='black')
        plt.legend(loc='best')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Sales')
        plt.title('Cumulative Probability Density Over Time')
        plt.show()

    def summary(self):
        print(self.res.summary())
        print('-' * 68)
        print('\n{:<24}{:<10}'.format('Term', 'Est'))

        print('\n{:<24}{:<10.5f}'.format('Coef. of Innovation', self.p))

        print('\n{:<24}{:<10.5f}'.format('Coef. of Imitation', self.q))

        print('\n{:<24}{:<10.2f}'.format('Peak Adoption Time', (np.log(self.q) - np.log(self.p) / (self.p + self.q))))

        print('\n{:<24}{:<10.2f}'.format('Max. Adopters', self.m))

        print('-' * 68)


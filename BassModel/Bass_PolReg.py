import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
from Bass import Bass


class Bass_PolReg(Bass):
    """
    this method uses non-linear least squares in order to estimate its coefficients. However, It's Levenberg-Marquadt nonlinear fitting for unbounded problems and a trust-region variant when bounds are given.
    we fit a curve into our BassModel function then try to estimate the best parameters for them
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
            self.sales = self.data[:, 1]
            self.time = range(1, len(self.sales) + 1)
            self.title = filename[: 2]

        except IndexError:

            print('Error: incorrect file format. Please provide either csv or txt.')

        self.popt = None
        self.pcov = None
        self.pred_sales = None
        self.sales_forecast = None
        self.cumsum_sales = None
        self.diff_y_pred = None
        self.q = None
        self.p = None
        self.m = None
        self.perr = None
        self.residual = None

    def __bass_model__(self, t, p, q, k):
        # print(len(self.time))
        np.seterr(over='ignore', invalid='ignore')
        self.residual = k * ((1 - np.exp(-1 * (p + q) * t)) / (1 + (q / p) * np.exp(-1 * (p + q) * t)))
        return self.residual

    def fit(self):
        """
         Method to fit the data to the Bass Model for Levenberg-Marquadt nonlinear
         Returns
         -------
         float
            coefficient of imitation, innovation, and the maximum number of adopters
         """
        self.popt, self.pcov = curve_fit(self.__bass_model__, self.time,
                                         self.sales)  # using non-linear least squares to fit a function for our data
        self.p = self.popt[0] #innovation
        self.q = self.popt[1] #imitation
        self.m = self.popt[2] #max potential
        # print(self.innovation, self.imitation, self.maximum_potential)
        return self.p, self.q, self.m

    def predict(self):
        self.pred_sales = self.__bass_model__(self.time, *self.popt)  # optimizing popt so that the sum of least squares is minimized
        self.sales_forecast = self.__bass_model__(np.linspace(1, len(self.time), 1000), *self.popt)

        return

    def plot(self):
        """
        PDF visualization of the predicted sales and actual sales
        Returns
        -------
        scatterplot of actual sales and regression of predicted sales
        """
        plt.plot(self.time, self.sales, 'o', label='Actual Sales', color='black')
        plt.plot(np.linspace(-1, max(self.time), 1000), self.sales_forecast, label='Sales Forecast')
        plt.legend(loc='best')
        plt.xlabel('Time')
        plt.ylabel('Sales')
        plt.title('Bass Model for Product Sales')

        plt.show()

    def plot_cdf(self):
        """
        CDF visualization of the predicted sales
        Returns
        -------
        Regression of cumulative predicted sales
        """
        self.cumsum_sales = np.cumsum(self.sales_forecast)
        plt.plot(np.linspace(-1, max(self.time), 1000), self.cumsum_sales, label='CProb',
                 color='black')
        plt.legend(loc='best')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Probability Density Over Time')
        plt.show()

    def summary(self):

        print('-' * 68)
        print('Bass Diffusion Model Diagnostics:\n')

        print('\n{:<24}{:<10}'.format('Metric',
                                      'Value'))

        print('\n{:<24}{:<10.3f}'.format('RMSE', np.sqrt(mean_squared_error(self.sales, self.pred_sales))))

        print('\n{:<24}{:<10.4f}'.format('R-Squared', r2_score(self.sales, self.pred_sales)))

        print('\n{:<24}{:<10.4f}'.format('Adj. R-Squared', 1 - (
                (1 - r2_score(self.sales, self.pred_sales)) * (len(self.time) - 1) / (len(self.time) - 3 - 1))))

        print('\n{:<24}{:<10.0f}'.format('Df',
                                         len(self.time) - 4))

        self.__error__()

        low_est = self.popt - self.perr
        up_est = self.popt + self.perr

        print('-' * 68)

        print('\n{:<24}{:<10}{:<10}{:<12}{:<12}'.format('Term', 'Est', 'Std Dev', 'Lower Lim', 'Upper Lim'))

        print('\n{:<24}{:<10.5f}{:<10.5f}{:<12.5f}{:<12.5f}'.format('Coef. of Innovation',
                                                                    self.popt[0],
                                                                    self.perr[0],
                                                                    low_est[0],
                                                                    up_est[0]))

        print('\n{:<24}{:<10.5f}{:<10.5f}{:<12.5f}{:<12.5f}'.format('Coef. of Imitation',
                                                                    self.popt[1],
                                                                    self.perr[1],
                                                                    low_est[1],
                                                                    up_est[1]))

        print('\n{:<24}{:<10.2f}{:<10}{:<12.2f}{:<12.2f}'.format('Peak Adoption Time',
                                                                 (np.log(self.popt[1]) - np.log(self.popt[0])) /
                                                                 (self.popt[0] + self.popt[1]),
                                                                 '',
                                                                 (np.log(low_est[1]) - np.log(low_est[0])) /
                                                                 (low_est[0] + low_est[1]),
                                                                 (np.log(up_est[1]) - np.log(up_est[0])) /
                                                                 (up_est[0] + up_est[1])))

        print('\n{:<24}{:<10.2f}{:<10.3f}{:<12.2f}{:<12.2f}'.format('Max. Adopters',
                                                                    self.popt[2],
                                                                    self.perr[2],
                                                                    low_est[2],
                                                                    up_est[2]))

        print('-' * 68)

    def __error__(self, cumu=True):
        p, q, k = self.popt - self.perr

        t = np.linspace(-1, max(self.time), 1000)

        self.diff_y_pred_lower = (k * p * (p + q) ** 2 * np.exp(t * (p + q))) / (
                (p * np.exp(t * (p + q)) + q) ** 2)

        p, q, k = self.popt + self.perr

        self.diff_y_pred_upper = (k * p * (p + q) ** 2 * np.exp(t * (p + q))) / (
                (p * np.exp(t * (p + q)) + q) ** 2)


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


class EnvelopeDetection:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self._get_right_edge_points()
        self._get_regression()
        self._get_bottom_regression()

    def _get_right_edge_points(self, percentile=99.8):##99.5
        _, bin_edges = pd.qcut(self.y, q=10, retbins=True, duplicates='drop') #changed q=10 to 6 MI

        x_edge_list = []
        y_edge_list = []
        for i in range(len(bin_edges)-1):
            idx_bin = np.where(np.logical_and(self.y >= bin_edges[i], self.y<bin_edges[i+1]))
            x_val = np.percentile(np.take(self.x, idx_bin), percentile)
            y_val = np.median(np.take(self.y, idx_bin))

            if not np.isnan(x_val) and not np.isnan(y_val):
                x_edge_list.append(x_val)
                y_edge_list.append(y_val)

        x_edge_list = np.array(x_edge_list)
        y_edge_list = np.array(y_edge_list)

        x_max_y = x_edge_list[np.argmax(y_edge_list)] #Adding condition to ignore all x having y less than the max y (we need values after triangle's top) MI

        print('x_max_y: ', x_max_y)

        self.binss =  np.array(bin_edges)
        self.x_edge = x_edge_list[x_edge_list>=x_max_y]
        self.y_edge = y_edge_list[np.where(x_edge_list>=x_max_y)]
        print('self.x_egde: ',self.x_edge)
        print('self.y_egde: ',self.y_edge)

    def _get_bottom_regression(self, percentile=99.8):
        y_bottom_val = np.percentile(self.y[np.logical_and(~np.isnan(self.y),~(self.y==0))], 100-percentile)
        print('y_bottom_val: ',y_bottom_val)

        self.slope_bottom = 0
        self.intercept_bottom = y_bottom_val

    def _get_regression(self):
        reg = LinearRegression().fit(self.x_edge.reshape(-1, 1), self.y_edge)

        self.slope = reg.coef_[0]
        self.intercept = reg.intercept_

    def show_scatter(self):

        fig, axis = plt.subplots()

        axis.set_title('Temperature vs. NDVI', fontsize=10)
        axis.set_ylabel('Temperature (Celsius)', fontsize=10)
        axis.set_xlabel('NDVI', fontsize=10)

        axis.scatter(self.x, self.y, s=2, c='blue')# c = toPlot['labels'].map(colors))
        axis.scatter(self.x_edge, self.y_edge, s=15, marker='^', c='red')
        axis.scatter(self.binss*0, self.binss, s=15, marker='^', c='black')

        x_line = np.linspace(0.75, 1, num=5) #made start=0.5 to make it similar to smi_estimation.py, then to 0.85 to ignore lower values messing with regression MI
        #x_line = np.linspace(0.7, 1, num=5)
        print('Envelope Detection')
        print('xline: ', x_line)
        print('slope: ', self.slope)
        print('interc: ', self.intercept)
        print('y: ', x_line * self.slope + self.intercept)
        axis.plot(x_line, (x_line * self.slope) + self.intercept, color='black')
        #axis.plot(self.x_edge, self.y_edge, color='red') #Test

        x_line = np.linspace(-1, 1, num=5)
        axis.plot(x_line, (x_line * self.slope_bottom) + self.intercept_bottom, color='black')

        plt.show()

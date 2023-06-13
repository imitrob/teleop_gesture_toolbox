import sys
import matplotlib
matplotlib.use('Qt5Agg')

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from collections import deque
import numpy as np

'''
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10, 10, 0.01)
y = x**2

#adding text inside the plot
plt.text(-5, 60, 'Parabola $Y = x^2$', fontsize = 22)

plt.plot(x, y, c='g')

plt.xlabel("X-axis", fontsize = 15)
plt.ylabel("Y-axis",fontsize = 15)

plt.show()
-------------------------------------------------------
import matplotlib.pyplot as plt

x = [1000, 2000, 3000, 4000 ,5000]
y = [1, 4, 9, 6, 10]

fig, ax = plt.subplots()

# instanciate a figure and ax object
# annotate is a method that belongs to axes
ax.plot(x, y, 'ro',markersize=23)

## controls the extent of the plot.
offset = 1.0
ax.set_xlim(min(x)-offset, max(x)+ offset)
ax.set_ylim(min(y)-offset, max(y)+ offset)

# loop through each x,y pair
for i,j in zip(x,y):
    ax.annotate(str(j),  xy=(i, j), color='white',
                fontsize="large", weight='heavy',
                horizontalalignment='center',
                verticalalignment='center')
    print(i, j)

plt.show()
'''
class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111, xlabel='time [s]', ylabel='Total timewarp distance [mm]')
        super(MplCanvas, self).__init__(fig)

class AnotherWindow(QWidget):

    def __init__(self, *args, **kwargs):
        super(QWidget, self).__init__(*args, **kwargs)

        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        max_len = 5
        self.xdata = deque(maxlen=max_len)
        self.ydata = deque(maxlen=max_len)

        # We need to store a reference to the plotted line
        # somewhere, so we can apply the new data to it.
        #self._plot_refs = None

        self.show()


    def set_n_series(self, n_series):
        self.n_series = n_series
        #self._plot_refs = [None] * n_series

    def update_plot(self, stamp, values):
        # Drop off the first y element, append a new one.

        self.xdata.append(stamp)
        self.ydata.append(values)
        ydata = np.array(self.ydata)

        self.canvas.axes.cla()  # clear the axes content
        self.canvas.axes.plot(self.xdata, ydata)
        self.canvas.draw_idle()
        self.canvas.axes.text(0, 0, 'Parabola $Y = x^2$', fontsize = 22)
        return
        # Note: we no longer need to clear the axis.
        if self._plot_refs[0] is None:
            # First time we have no plot reference, so do a normal plot.
            # .plot returns a list of line <reference>s, as we're
            # only getting one we can take the first element.
            for i in range(self.n_series):
                self._plot_refs[i], = self.canvas.axes.plot(self.xdata, ydata[:,0], self.colors[i])
        else:
            # We have a reference, we can use it to update the data for that line.
            for i in range(self.n_series):
                self._plot_refs[i], = self.canvas.axes.plot(self.xdata, ydata[:,0], self.colors[i])
                #self._plot_refs[i].set_ydata(ydata[:,0])

        # Trigger the canvas to update and redraw.
        self.canvas.draw()

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.plot_window = None

        self.button = QPushButton("Push for Window")
        self.button.clicked.connect(self.show_new_window)
        self.setCentralWidget(self.button)

    def show_new_window(self, checked):
        if self.plot_window is None:
            self.plot_window = AnotherWindow()
            self.plot_window.show()
            self.plot_window.set_n_series(5)

            self.plot_window.update_plot(0, [3,4,5,6,7])
            self.plot_window.update_plot(1, [4,5,6,7,8])
            self.plot_window.update_plot(2, [5,6,7,8,9])
            self.plot_window.update_plot(3, [5,6,7,8,9])
            self.plot_window.update_plot(5, [5,6,7,8,9])
            self.plot_window.update_plot(6, [2,2,2,2,2])
            self.plot_window.update_plot(7, [2,2,2,2,2])
            print(self.plot_window.xdata)
            print(self.plot_window.ydata)
        else:
            self.plot_window = None  # Discard reference, close window.



app = QApplication(sys.argv)
w = MainWindow()
w.show()
app.exec()

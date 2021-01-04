
import os
import sys
import math
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtGui, QtCore
from gui import Ui_MainWindow


def read_file(file_name):
    file_path = os.path.join(os.getcwd(), file_name)
    with open(file_path, 'r') as f:
        dataset = np.array([lines.strip('\n').split() for lines in f], dtype=float)
    dim1 = dataset.shape[1] - 1
    X = dataset[:, :dim1]
    Y = dataset[:, dim1]
    Y = np.array(Y, dtype=int)
    X, Y = shuffle(X, Y)
    return (X, Y)

def shuffle(X, Y):
    randomize = np.arange(0, len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def plot_weight_and_inputs(ax, X, Y, W, iters):
    if X.shape[1] > 2:
        return

    elif X.shape[1] == 2:
        plt.cla() 
        plt.ion()  
        ax.scatter(X[:, 0], X[:, 1], color=plt.cm.Set1(Y))
        ax.scatter(W[:, :, 0], W[:, :, 1], color='r')
        plt.title(f'SOM Topology Map epochs:{iters+1}')
        plot_network(W)
        plt.pause(0.2)
        plt.show()
        

def plot_network(W):
    G = nx.Graph()
    idx = 1
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            G.add_node(idx, pos=W[i, j])
            idx += 1

    for idx in range(1, W.shape[0]*W.shape[1]+1):
        if idx % W.shape[0] == 0:
            pass
        else:
            G.add_edge(idx, idx+1)
        if idx > W.shape[0]*(W.shape[1]-1):
            pass
        else:
            G.add_edge(idx, idx+W.shape[0])

    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, node_color='red', node_size=10)
    plt.axis("on")
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.show()


def plot_feature_map(feature_map):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111) 
    plt.imshow(feature_map, cmap='bone')
    plt.colorbar()
    plt.title('SOM Feature Map')
    plt.show()

class SOM:
    def __init__(self, X, neuron_size, iters):
        self.X = X
        self.neuron_size = neuron_size
        self.weights = np.random.rand(neuron_size[0], neuron_size[1], X.shape[1])
        self.lr_const = 0.5
        self.radius_const = 1
        self.iters = iters     

    def normalize_X(self, X):
        for i in range(X.shape[1]):
            X_max = np.max(X[:, i])
            X_min = np.min(X[:, i])
            X_av = np.average(X[:, i])
            if X_max-X_min == 0:
                X[:, i] = 0
            else:
                X[:, i] = (X[:, i]-X_av) / (X_max-X_min)

        return X

    def get_lr(self, iters):
        return self.lr_const * np.exp(-iters / self.iters)

    def get_radius(self, iters):
        return self.radius_const * np.exp(-iters / self.iters)

    def get_BMU(self, X):
        dist_mat = np.zeros([self.X.shape[1]])
        dist = 0
        loc = [0, 0]
        for i in range(self.neuron_size[0]):
            for j in range(self.neuron_size[1]):
                dist_mat = X - self.weights[i, j]
                if i == 0 and j ==0:
                    dist = np.sum(dist_mat**2)
                else:
                    if np.sum(dist_mat**2) < dist:
                        dist = np.sum(dist_mat**2)
                        loc = [i, j]

        return loc

    def update_weights(self, X, Y):
        if X.shape[1] == 2:
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111) 
        for iters in range(self.iters):
            if X.shape[1] == 2:
                plot_weight_and_inputs(ax, X, Y, self.weights, iters)
            for x in self.normalize_X(self.X):
                BMU_loc = self.get_BMU(x)
                lr = self.get_lr(iters)
                radius = self.get_radius(iters)
                for i in range(self.neuron_size[0]):
                    for j in range(self.neuron_size[1]):
                        neighbor_dist = np.exp(-((BMU_loc[0]-i)**2 + (BMU_loc[1]-j)**2) / (2 * radius**2))
                        self.weights[i, j] += lr * neighbor_dist * (x-self.weights[i, j])
    
    def get_feature_map(self):
        dist_map = np.zeros([self.neuron_size[0], self.neuron_size[1]])
        for i in range(self.neuron_size[0]):
            for j in range(self.neuron_size[1]):
                neighbor_dist = 0
                if i+1 < self.neuron_size[0]:
                        neighbor_dist += np.sqrt(np.sum((self.weights[i, j]-self.weights[i+1, j])**2))
                if i-1 >= 0:
                        neighbor_dist += np.sqrt(np.sum((self.weights[i, j]-self.weights[i-1, j])**2))
                if j+1 < self.neuron_size[1]:
                        neighbor_dist += np.sqrt(np.sum((self.weights[i, j]-self.weights[i, j+1])**2))
                if j-1 >= 0:
                        neighbor_dist += np.sqrt(np.sum((self.weights[i, j]-self.weights[i, j-1])**2))
                dist_map[i, j] = neighbor_dist
  
        return dist_map
    
    def get_weights(self):
        return self.weights

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.train)
        self.ui.pushButton_2.clicked.connect(self.choose_file)
        self.ui.plainTextEdit_2.setFont(QtGui.QFont('Arial', 8))
        self.ui.plainTextEdit.setFont(QtGui.QFont('Arial', 14))

    def train(self):
        filename = self.ui.plainTextEdit_2.toPlainText()
        iters = int(self.ui.plainTextEdit.toPlainText())

        X, Y = read_file(filename)
        if len(X) > 1000:
            size = 8
        else:
            size = math.ceil(np.sqrt(5 * np.sqrt(len(X))))

        som = SOM(X, neuron_size=[size, size], iters=iters)

        som.update_weights(X, Y)
        plot_feature_map(som.get_feature_map())

    def choose_file(self):
        options = QtWidgets.QFileDialog.Options()  # 開啟選單
        options |= QtWidgets.QFileDialog.DontUseNativeDialog  # 使用非本機Dialog
        filename, filetype = QtWidgets.QFileDialog.getOpenFileName(self, 
                                                        "Choosing file...", 
                                                        "./",
                                                        "All Files (*);;Python Files (*.py)",
                                                        options=options)  # 參數1:Dialog Title, 參數2:起始路徑 ,參數3:檔案類型
        self.ui.plainTextEdit_2.setPlainText(filename)

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())





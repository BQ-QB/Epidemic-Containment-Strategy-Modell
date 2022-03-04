import flask
import os
from flask import send_from_directory, render_template
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np

n = 80
l = 10
initial_infected = 5
x = np.floor(np.random.rand(n) * l)  # x coordinates
y = np.floor(np.random.rand(n) * l)  # y coordinates
S = np.zeros(n)  # status array, 0: Susceptiple, 1: Infected, 2: recovered, 3: Dead
S[0:initial_infected] = 1
D = 0.8

app = flask.Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/plot/')
def plot():
    img = io.BytesIO()
    
    sus = np.where(S==0)[0]
    x_sus = []
    y_sus = []
    for index in sus:
        x_sus.append(x[index])
        y_sus.append(y[index])

    x_inf = []
    y_inf = []
    inf = np.where(S==1)[0]
    for index in inf:
        x_inf.append(x[index])
        y_inf.append(y[index])
    
    x_rec = []
    y_rec = []
    rec = np.where(S==2)[0]
    for index in rec:
        x_rec.append(x[index])
        y_rec.append(y[index])

    fix, ax = plt.subplots(figsize=(5,5))

    ax.scatter(x_sus,y_sus, color='blue')
    ax.scatter(x_inf,y_inf, color='red')

    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    plt.savefig(img,format='png')

    img.seek(0)

    plot_url = base64.b64encode(img.getvalue()).decode()

    return '<img src="data:image/png;base64,{}">'.format(plot_url)

def update_position():
    steps_x_or_y = np.random.rand(n)
    steps_x = steps_x_or_y < D / 2
    steps_y = (steps_x_or_y > D / 2) & (steps_x_or_y < D)
    nx = (x + np.sign(np.random.randn(n)) * steps_x) % l
    ny = (y + np.sign(np.random.randn(n)) * steps_y) % l
    return nx, ny

if __name__ == "__main__":
    app.secret_key = 'Epidemic_Enjoyers'
    app.debug = True
    app.run()
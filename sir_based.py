import numpy as np 
from tkinter import *
import matplotlib.pyplot as plt

res = 500   # Animation resolution
tk = Tk()  
tk.geometry(str(int(res*1.1)) + 'x' + str(int(res*1.3)))
tk.configure(background='white')

canvas = Canvas(tk, bd=2)            # Generate animation window 
tk.attributes('-topmost', 0)
canvas.place(x=res/20, y=res/20, height=res, width=res)
ccolor = ['#0008FF', '#DB0000', '#12F200', '#68228B', '#000000']


def restart():
    global S
    I = np.argsort((x-l/2)**2 + (y-l/2)**2)
    S = np.zeros(n) 
    S[I[1:initial_infected]] = 1


def plot_sir():
    temp1 = SH.shape[0]
    temp1 = np.array([i for i in range(temp1)])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    label1 = 'Susceptible = ' + str(SH[-1])
    label2 = 'Recovered = ' + str(RH[-1])
    label3 = 'Infected = ' + str(IH[-1])
    label4 = 'Dead = ' + str(DH[-1])
    ax.plot(temp1, SH, color='yellow', label=label1)
    ax.plot(temp1, RH, color='blue', label=label2)
    ax.plot(temp1, IH, color='red', label=label3)
    ax.plot(temp1, DH, color='purple', label=label4)
    ax.set_title('Infection plot')
    ax.legend()
    plt.show()


showPlot = Button(tk, text='Plot', command=plot_sir)
showPlot.place(relx=0.05, rely=0.85, relheight=0.06, relwidth=0.15)

rest = Button(tk, text='Restart', command=restart)
rest.place(relx=0.05, rely=.91, relheight=0.06, relwidth=0.15)

Beta = Scale(tk, from_=0, to=1, orient=HORIZONTAL, label='Infection probability', font=("Helvetica", 8), resolution=0.01)
Beta.place(relx=.22, rely=.85, relheight=0.12, relwidth=0.23)
Beta.set(1)            # Parameter slider for infection rate                                                       

Gamma = Scale(tk, from_=0, to=0.1, orient=HORIZONTAL, label='Recovery rate', font=("Helvetica", 8), resolution=0.001)
Gamma.place(relx=.47, rely=.85, relheight=0.12, relwidth=0.23)
Gamma.set(0.01)          # Parameter slider for recovery rate

Diff = Scale(tk, from_=0, to=1, orient=HORIZONTAL, label='Diffusion probability', font=("Helvetica", 8), resolution=0.01)
Diff.place(relx=.72, rely=.85, relheight=0.12, relwidth=0.23)
Diff.set(0.5)            # Parameter slider for diffusion rate


# Parameters of the simulation
n = 800       # Number of agents 
initial_infected = 4   # Initial infected agents
N = 100000  # Simulation time
l = 50     # Lattice size

# Historylists used for plotting SIR-graph
IH = np.array([initial_infected-1])
SH = np.array([n-initial_infected+1])
RH = np.array([0])
DH = np.array([0])

#Contact matrix
contact = -1*np.ones((n, 5))


# Physical parameters of the system 
x = np.floor(np.random.rand(n)*l)          # x coordinates            
y = np.floor(np.random.rand(n)*l)          # y coordinates  
S = np.zeros(n)                            # status array, 0: Susceptiple, 1: Infected, 2: recovered, 3: Dead 
isolated = np.zeros(n)                     # Isolation array, 0: not isolated, 1: Is currently in isolation                   # test array; 0: Should not be isolated, 1: Positive test, should be isolated 
Q = np.zeros(n)                            # temperature array
I = np.argsort((x-l/2)**2 + (y-l/2)**2)
S[1:initial_infected] = 1              # Infect agents that are close to center 

nx = x                           # updated x                  
ny = y                           # updated y  

lockdown = True

particles = []
R = .5                          # agent plot radius 
for j in range(n):     # Generate animated particles in Canvas 
    particles.append(canvas.create_oval((x[j])*res/l,
                                         (y[j])*res/l,
                                         (x[j]+2*R)*res/l,
                                         (y[j]+2*R)*res/l,
                                         outline=ccolor[0], fill=ccolor[0]))


def set_temps():
    for i in np.where(S == 1)[0]:
        Q[i] = np.random.normal(40,1)
        print(Q[i])
    for i in np.where(Q == 0)[0]:
        Q[i] = np.random.normal(36,1)


t = 0
set_temps()
B = Beta.get()
G = Gamma.get()
D = Diff.get()
My = 0.00


while t < 1000 and list(np.where(S == 1)[0]):

    steps_x_or_y = np.random.rand(n)
    steps_x = steps_x_or_y < D/2
    steps_y = (steps_x_or_y > D/2) & (steps_x_or_y < D)
    nx = (x + np.sign(np.random.randn(n)) * steps_x) % l 
    ny = (y + np.sign(np.random.randn(n)) * steps_y) % l
    for i in np.where(((isolated != 0) | (S == 3))):
        nx[i] = x[i]
        ny[i] = y[i]

    for i in np.where((isolated != 1) & (S == 1) & (np.random.random(n) < B))[0]:     # loop over infecting agents
        Q[(x == x[i]) & (y == y[i]) & (S == 0)] = np.random.normal(40, 1)          # Raise newly sick agents temperatures
        S[(x == x[i]) & (y == y[i]) & (S == 0)] = 1         # Susceptiples together with infecting agent becomes infected

    for i in np.where((S == 1) & (np.random.random(n) < My))[0]:
        S[i] = 3

    temp_list = np.where((S == 1) & (np.random.rand(n) < G))[0]
    S[temp_list] = 2         # Recovery
    # Isolated[ templist ] = 0
    Q[temp_list] = np.random.normal(36, 1)

    for j in range(n):
        canvas.move(particles[j], (nx[j]-x[j]) * res/l, (ny[j]-y[j])*res/l)         # Plot update - Positions
        canvas.itemconfig(particles[j], outline='#303030', fill=ccolor[int(S[j]) if isolated[j] == 0 else 4])  # Plot update - Colors
    tk.update()
    tk.title('Infected:' + str(np.sum(S==1)))

    # Management of contactmatrix
    for i in range(n):

        proximity_list = np.where((x == x[i]) & (y == y[i]))
        
        for j in range(min(5, len(proximity_list[0]))):
            contact[i][j] = proximity_list[0][j]
           


    # Tests sick agents, if positive test then set in isolation
    if t > 5:

        test_capacity = 10
        test_priority = np.argsort(Q)
        
        i = 0
        while i < test_capacity:
            if isolated[test_priority[n-1-i]] != 1:
                if int(S[test_priority[n-1-i]]) == 1:
                    isolated[test_priority[n-i-1]] = 1
                    for k in range(5):
                        if contact[test_priority[n-i-1]][k] != -1:
                            isolated[int(contact[test_priority[n-i-1]][k])] = 1
                            print(int(contact[test_priority[n-i-1]][k]))

            i = i+1


    # lockdown loop
    startLock = 50
    if startLock < t < startLock + 200 and lockdown:
        D = 0.1
    else: D = Diff.get()

    x = nx                                              # Update x 
    y = ny                                              # Update y 

    SH = np.append(SH, len(list(np.where(S == 0)[0])))
    IH = np.append(IH, len(list(np.where(S == 1)[0])))
    RH = np.append(RH, len(list(np.where(S == 2)[0])))
    DH = np.append(DH, len(list(np.where(S == 3)[0])))
    t += 1

    if t % 300 == 0:
        plot_sir()

Tk.mainloop(canvas)                                     # Release animation handle (close window to finish) 
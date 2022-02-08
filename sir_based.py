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


show_plot = Button(tk, text='Plot', command=plot_sir)
show_plot.place(relx=0.05, rely=0.85, relheight=0.06, relwidth=0.15)



# Parameters of the simulation
n = 1000     # Number of agents 
initial_infected = 50   # Initial infected agents
N = 100000  # Simulation time
l = 80     # Lattice size

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
temperatures  = np.zeros(n)                            # temperature array
S[1:initial_infected] = 1              # Infect agents that are close to center 

nx = x                           # updated x                  
ny = y                           # updated y  

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
        temperatures [i] = np.random.normal(40,1)

    for i in np.where(temperatures  == 0)[0]:
        temperatures [i] = np.random.normal(36,1)
        
#test

# Modifiable parameters by the user 

D_noll = 0.8
D_reduced = 0.1

D = D_noll
B = 1
G = 0.03

My = 0.00
start_lock = 50
lockdown_enabled = True
test_capacity = 100

set_temps()
t = 0

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
        temperatures [(x == x[i]) & (y == y[i]) & (S == 0)] = np.random.normal(40, 1)          # Raise newly sick agents temperatures
        S[(x == x[i]) & (y == y[i]) & (S == 0)] = 1         # Susceptiples together with infecting agent becomes infected

    for i in np.where((S == 1) & (np.random.random(n) < My))[0]:
        S[i] = 3

    recovered_list = np.where((S == 1) & (np.random.rand(n) < G))[0]
    S[recovered_list] = 2         # Recovery
    # Isolated[ recovered_list ] = 0
    # temperatures [recovered_list] = np.random.normal(36, 1)

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
           

    
    
    # Tests sick agents, if positive test then set in isolation and isolate neighbours in contactmatrix
    if t > 5:
        
        test_priority = np.argsort(temperatures)
        i = 0
        tests_made = 0
        while tests_made < test_capacity and i<n-1 :
            if isolated[test_priority[-i-1]] != 1:
                
                if S[test_priority[-i-1]] == 1:
                    isolated[test_priority[-i-1]] = 1
                    for k in range(5):
                        if contact[test_priority[n-i-1]][k] != -1:
                            isolated[ int(contact[test_priority[n-i-1]][k]) ] = 1
                            
                tests_made += 1   

            i = i+1
        

  
    # lockdown_enabled loop
    if start_lock < t < start_lock + 200 and lockdown_enabled:
        D = D_reduced
    else: D = D_noll

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
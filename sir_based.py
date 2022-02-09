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



# Plots graph
def plot_sir():

    index_list_for_plot = susceptible_history.shape[0]
    index_list_for_plot = np.array([i for i in range(index_list_for_plot)])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    label_susceptible = 'Susceptible = ' + str(susceptible_history[-1])
    label_recovered = 'Recovered = ' + str(recovered_history[-1])
    label_infected = 'Infected = ' + str(infected_history[-1])
    label_dead = 'Dead = ' + str(dead_history[-1])
    label_isolation = 'Isolation = ' + str(isolation_history[-1])
    ax.plot(index_list_for_plot, susceptible_history, color = 'blue', label = label_susceptible)
    ax.plot(index_list_for_plot, recovered_history, color = 'green', label = label_recovered)
    ax.plot(index_list_for_plot, infected_history, color = 'red', label = label_infected)
    ax.plot(index_list_for_plot, dead_history, color = 'purple', label = label_dead)
    ax.plot(index_list_for_plot, isolation_history, color = 'black', label = label_isolation)
    ax.set_title('Infection plot')
    ax.legend()
    plt.show()


# Button to press to show the graph
show_plot = Button(tk, text='Plot', command=plot_sir)
show_plot.place(relx=0.05, rely=0.85, relheight=0.06, relwidth=0.15)



# Parameters of the simulation
n = 1000     # Number of agents 
initial_infected = 50   # Initial infected agents
N = 100000  # Simulation time
l = 80     # Lattice size

# Historylists used for plotting SIR-graph
infected_history = np.array([initial_infected-1])
susceptible_history = np.array([n-initial_infected+1])
recovered_history = np.array([0])
dead_history = np.array([0])
isolation_history = np.array([0])

#Contact matrix
contact = np.zeros((50, 1, n))


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

# sets initial temperatures of agents
def set_temps():
    for i in np.where(S == 1)[0]:
        temperatures [i] = np.random.normal(40,1)

    for i in np.where(temperatures  == 0)[0]:
        temperatures [i] = np.random.normal(36,1)
        

# Modifiable parameters by the user 

D_noll = 0.8
D_reduced = 0.1

D = D_noll
B = 1
G = 0.03

My = 0.00
start_lock = 50
lockdown_enabled = True
test_capacity = 30

set_temps()
t = 0

while t < 1000 and list(np.where(S == 1)[0]):

    # Updates positions, if not dead or isolate
    steps_x_or_y = np.random.rand(n)
    steps_x = steps_x_or_y < D/2
    steps_y = (steps_x_or_y > D/2) & (steps_x_or_y < D)
    nx = (x + np.sign(np.random.randn(n)) * steps_x) % l 
    ny = (y + np.sign(np.random.randn(n)) * steps_y) % l
    for i in np.where(((isolated != 0) | (S == 3))):
        nx[i] = x[i]
        ny[i] = y[i]

    #Infect Neighbours
    for i in np.where((isolated != 1) & (S == 1) & (np.random.random(n) < B))[0]:     # loop over infecting agents
        temperatures [(x == x[i]) & (y == y[i]) & (S == 0)] = np.random.normal(40, 1)          # Raise newly sick agents temperatures
        S[(x == x[i]) & (y == y[i]) & (S == 0)] = 1         # Susceptiples together with infecting agent becomes infected

    # Agents death
    for i in np.where((S == 1) & (np.random.random(n) < My))[0]:
        S[i] = 3

    
    # Recovery
    recovered_list = np.where((S == 1) & (np.random.rand(n) < G))[0]
    S[recovered_list] = 2                                         
    # Isolated[ recovered_list ] = 0                              # Lets recovered people out from isolation 
    # temperatures [recovered_list] = np.random.normal(36, 1)


    # update positions of agents in the simulation-window
    for j in range(n):
        canvas.move(particles[j], (nx[j]-x[j]) * res/l, (ny[j]-y[j])*res/l)         # Plot update - Positions
        canvas.itemconfig(particles[j], outline='#303030', fill=ccolor[int(S[j]) if isolated[j] == 0 else 4])  # Plot update - Colors
    tk.update()
    tk.title('Infected:' + str(np.sum(S==1)))

    
    # Management of contactmatrix, to be developed further to enhance performance
    # Currently a n x 5 matrix where the indexes of the last 5 contacts are saved
    prod_list = np.zeros(n)
    for i in range(n):

        prod_list[i] = (2**x[i])*(3**y[i])
    
    contact[t % 50] = prod_list
        
        
    # Tests sick agents, if positive test then set in isolation and isolate neighbours in contactmatrix
    if t > 50:
        
        test_priority = np.argsort(temperatures) # test_priority is an array of indexes corresponding to increasing temperatures
        
        i = 0
        tests_made = 0
        while tests_made < test_capacity and i<n-1 : # can't use more tests than allowed, and can't test more agents than there are agents
            if isolated[test_priority[-i-1]] != 1:   # Proceed if the selected agent is not already isolated
                tests_made += 1                      # A test is counted
                test_person = test_priority[-i-1]
                if S[test_person] == 1:      # If the agent is sick put them in isolation, and isolate the recent contacts
                    isolated[test_person] = 1 
                    test_person_coordinate = (2**x[test_person])*(3**y[test_person])

                    for k in range(min(50,t)):
                        for p in np.where(contact[(t-k)%50] == test_person_coordinate): 
                            if len(p)>0:             
                                isolated[p] = 1
                            
            i = i+1
        
    print(len(list(np.where(isolated==1)[0])))
  
    # lockdown_enabled loop
    if start_lock < t < start_lock + 200 and lockdown_enabled:
        D = D_reduced
    else: D = D_noll

    x = nx                                              # Update x 
    y = ny                                              # Update y 

    # Used for plotting the graph
    susceptible_history = np.append(susceptible_history, len(list(np.where(S == 0)[0])))
    infected_history = np.append(infected_history, len(list(np.where(S == 1)[0])))
    recovered_history = np.append(recovered_history, len(list(np.where(S == 2)[0])))
    dead_history = np.append(dead_history, len(list(np.where(S == 3)[0])))
    isolation_history = np.append(isolation_history, len(list(np.where(isolated == 1)[0])))
    
    t += 1

    if t % 300 == 0:
        plot_sir()

Tk.mainloop(canvas)                                     # Release animation handle (close window to finish) 
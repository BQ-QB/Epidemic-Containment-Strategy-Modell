import numpy as np 
from tkinter import *
from tkinter import ttk
import matplotlib.pyplot as plt

res = 500   # Animation resolution
tk = Tk()  
tk.geometry( str(int(res*1.1)) + 'x'  +  str(int(res*1.3)) )
tk.configure(background='white')

canvas = Canvas(tk, bd=2)            # Generate animation window 
tk.attributes('-topmost', 0)
canvas.place(x=res/20, y=res/20, height= res, width= res)
ccolor = ['#0008FF', '#DB0000', '#12F200', '#000000']

def restart():
    global S
    I = np.argsort((x-l/2)**2 + (y-l/2)**2)
    S = np.zeros(n) 
    S[I[1:initial_infected]] = 1
    
def plotSir():
    temp1 = SH.shape[0]
    temp1 = np.array([i for i in range(temp1)])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    label1 = 'Susceptible = ' + str(SH[-1])
    label2 = 'Recovered = ' + str(RH[-1])
    label3 = 'Infected = ' + str(IH[-1])
    ax.plot(temp1, SH, color='yellow', label = label1 )
    ax.plot(temp1, RH, color='blue', label = label2 )
    ax.plot(temp1, IH, color = 'red', label = label3 )
    #ax.plot(tl, d, color = 'black', label = 'Dead')
    ax.set_title('Infection plot')
    ax.legend()
    plt.show()


showPlot = Button(tk, text = 'Plot', command = plotSir)
showPlot.place(relx = 0.05, rely = 0.85, relheight=0.06, relwidth=0.15)

rest = Button(tk, text='Restart',command= restart) 
rest.place(relx=0.05, rely=.91, relheight= 0.06, relwidth= 0.15 )

Beta = Scale(tk, from_=0, to=1, orient=HORIZONTAL, label='Infection probability', font=("Helvetica", 8),resolution=0.01)
Beta.place(relx=.22, rely=.85, relheight= 0.12, relwidth= 0.23)     
Beta.set(1)            # Parameter slider for infection rate                                                       

Gamma = Scale(tk, from_=0, to=0.1, orient=HORIZONTAL, label='Recovery rate', font=("Helvetica", 8) ,resolution=0.001)
Gamma.place(relx=.47, rely=.85, relheight= 0.12, relwidth= 0.23)
Gamma.set(0.01)          # Parameter slider for recovery rate

Diff = Scale(tk, from_=0, to=1, orient=HORIZONTAL, label='Diffusion probability', font=("Helvetica", 8),resolution=0.01)
Diff.place(relx=.72, rely=.85, relheight= 0.12, relwidth= 0.23)
Diff.set(0.5)            # Parameter slider for diffusion rate


# Parameters of the simulation
n = 800          # Number of agents 
initial_infected = 40  # Initial infected agents 
N = 100000  # Simulation time
l = 50     # Lattice size

# Historylists used for plotting SIR-graph
IH = np.array([initial_infected-1])
SH = np.array([n-initial_infected+1])
RH = np.array([0])

#Contact matrix
Contact = -1*np.ones((n,5))


# Physical parameters of the system 
x = np.floor(np.random.rand(n)*l)          # x coordinates            
y = np.floor(np.random.rand(n)*l)          # y coordinates  
S = np.zeros(n)                            # status array, 0: Susceptiple, 1: Infected, 2: recovered 
Isolated = np.zeros(n)                     # Isolation array, 0: not isolated, 1: Is currently in isolation
toBeTested = np.zeros(n)                 # test array; 0: Should not be isolated, 1: Positive test, should be isolated 
Q = np.zeros(n)                            # temperature array
I = np.argsort((x-l/2)**2 + (y-l/2)**2)
S[I[1:initial_infected]] = 1              # Infect agents that are close to center 

nx = x                           # udpated x                  
ny = y                           # updated y                  

particles = []
R = .5                          # agent plot radius 
for j in range(n):     # Generate animated particles in Canvas 
    particles.append( canvas.create_oval( (x[j] )*res/l,                                           
                                         (y[j] )*res/l,                                           
                                         (x[j]+2*R )*res/l,                                           
                                         (y[j]+2*R )*res/l,                                           
                                         outline=ccolor[0], fill=ccolor[0]) )
    
def setTemps(): 
    for i in np.where(S == 1)[0]:
        Q[i] = np.random.normal(40,1)
        print(Q[i])
    for i in np.where(Q == 0):
        Q[i] = np.random.normal(36,1)


t = 0
setTemps()

while t<1000 and list(np.where(S==1)[0]):
    
    B = Beta.get()
    G = Gamma.get()
    D = Diff.get()
    
    steps_x_or_y = np.random.rand(n)
    steps_x = steps_x_or_y < D/2
    steps_y = (steps_x_or_y > D/2) & (steps_x_or_y < D)
    nx = (x + np.sign(np.random.randn(n)) * steps_x) % l 
    ny = (y + np.sign(np.random.randn(n)) * steps_y) % l
    for i in np.where((Isolated == 1)):
        nx[i] = x[i]
        ny[i] = y[i]
    
    
    for i in np.where((Isolated != 1) & (S==1) & ( np.random.random(n) < B ))[0]:     # loop over infecting agents 
        Q[(x==x[i]) & (y==y[i]) & (S==0)] = np.random.normal(40,1)          # Raise newly sick agents temperatures
        S[(x==x[i]) & (y==y[i]) & (S==0)] = 1         # Susceptiples together with infecting agent becomes infected 
        
    templist = np.where((S==1) & (np.random.rand(n) < G))[0]
    S[ templist ] = 2         # Recovery
    # Isolated[ templist ] = 0
    # Q[templist] = np.random.normal(36,1)

    for j in range(n):
        canvas.move(particles[j], (nx[j]-x[j]) *res/l, (ny[j]-y[j])*res/l)         # Plot update - Positions 
        canvas.itemconfig(particles[j], outline='#303030', fill=ccolor[int(S[j]) if Isolated[j] == 0 else 3]) # Plot update - Colors  
    tk.update()
    tk.title('Infected:' + str(np.sum(S==1)))

    # Management of contactmatrix
    for i in range(n):
        
        proximitylist = (x == x[i]) & (y == y[i])
    
        for j in range(min(5, len(proximitylist))):
            Contact[i][j] = proximitylist[j]
    


    # Tests sick agents, if positive test then set in isolation
    if t>5:
        
        testCapacity = 10
        testPriority = np.argsort(Q)
        
        i = 0
        while i<testCapacity:
            if Isolated[testPriority[n-1-i]]!= 1:
                if S[testPriority[n-1-i]] == 1:
                    Isolated[testPriority[n-i-1]] = 1
                    for k in range(len(Contact[testPriority[n-i-1]])):
                        Isolated[ int(Contact[testPriority[n-i-1]][k]) ] = 1

                    
            i = i+1
    
    

    x = nx                                              # Update x 
    y = ny                                              # Update y 
    SH = np.append(SH, len(list(np.where(S ==0)[0])))
    IH = np.append(IH, len(list(np.where(S==1)[0])))
    RH = np.append(RH, len(list(np.where(S==2)[0])))
    t+=1
    if t %300 == 0:
        plotSir()
    
Tk.mainloop(canvas)                                     # Release animation handle (close window to finish)

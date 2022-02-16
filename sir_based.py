import numpy as np
from tkinter import *
import matplotlib.pyplot as plt
"""
import keras
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import np_utils #Needed to enable "to_categorical"

np.seterr(invalid='ignore')


def setupNN():
    model = Sequential()  # Define the NN model
    model.add(Dense(16, input_dim=5, activation='relu'))  # Add Layers
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='softmax'))  # softmax ensures number between 0-1.
    model.compile(loss='mean_squared_error', optimizer='adam', metrics='accuracy')
    return model


def trainNN(model, t):
    if t > 20:
        pass
        # Setup the training lists and feed them to the NN
        # Input för NN
        # arry/listan för y_train består av lång lista som korresponderar till x_train där varje index är 0 för frisk eller 1 för sjuk.
        # model.fit(x_train, y_train, epochs=100) #vilken batch size?  #Input för NN, lista, där varje plats är matrix som i artikeln
        # model.evaluate(x_test, y_test, verbose=1)

        resultNN = model.layers[
            3].output  # Output för NN, Behöver eventuellt ändra idex beroende på om dropout räknas som lager, vill få output från softmax
        # model.summary() Få tag i info om modellens uppbyggnad
        return resultNN


def deployNN(model, t):
    if t > 20:
        result = trainNN(model)

    for n in result:
        p = result[n]
        if p > 0.995:
            pass
            # isolate agent
            if 0.5 < p < 0.995:
                pass
                # add to test array and test 100 agents with the highest temperature
"""

def __init__():
    x = np.floor(np.random.rand(n) * l)  # x coordinates
    y = np.floor(np.random.rand(n) * l)  # y coordinates
    S = np.zeros(n)  # status array, 0: Susceptiple, 1: Infected, 2: recovered, 3: Dead
    isolated = np.zeros(n)  # Isolation array, 0: not isolated, 1: Is currently in isolation
    temperatures = np.zeros(n, dtype='float16')  # temperature array
    tested = np.zeros(n)
    S[0:initial_infected] = 1  # Infect random agents
    nx = x  # updated x
    ny = y  # updated y
    return x, y, S, isolated, temperatures, tested, nx, ny


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
    ax.plot(index_list_for_plot, susceptible_history, color='blue', label=label_susceptible)
    ax.plot(index_list_for_plot, recovered_history, color='green', label=label_recovered)
    ax.plot(index_list_for_plot, infected_history, color='red', label=label_infected)
    ax.plot(index_list_for_plot, dead_history, color='purple', label=label_dead)
    ax.plot(index_list_for_plot, isolation_history, color='black', label=label_isolation)
    ax.set_title('Infection plot')
    ax.legend()
    plt.show()


def update_position():
    steps_x_or_y = np.random.rand(n)
    steps_x = steps_x_or_y < D / 2
    steps_y = (steps_x_or_y > D / 2) & (steps_x_or_y < D)
    nx = (x + np.sign(np.random.randn(n)) * steps_x) % l
    ny = (y + np.sign(np.random.randn(n)) * steps_y) % l
    for i in np.where(((isolated != 0) | (S == 3)))[0]:
        nx[i] = x[i]
        ny[i] = y[i]
    return nx, ny


def gen_contacts():
    contact_list = np.zeros(n)
    sick_contact_list = np.zeros(n)

    coord_list = np.array([2**x[i] * 3**y[i] for i in range(n)])

    sick_free_agents = np.where((S == 1) & (isolated != 1))[0]
    non_dead_free_agents = np.where((S != 3) & (isolated != 1))[0]

    for infected in sick_free_agents :
        infected_agent = infected
        for other_agent in non_dead_free_agents:
            if (coord_list[infected_agent] == coord_list[other_agent]) & (infected_agent != other_agent):
                sick_contact_list[other_agent] += 1
            
    for i in range(n):
        for hits in np.where((x[i] == x) & (y[i] == y) & (isolated != 1))[0]:
            contact_list[i] += 1
    

    contact_i[t % 50] = sick_contact_list
    contact_tot[t % 50] = contact_list

    contact_q[t % 10] =  np.nan_to_num(np.divide(np.sum(contact_i, 0),np.sum(contact_tot, 0)))
    


def gen_R():  # Generatorfunktion för R-matriserna

    temp_r16 = np.zeros(n)
    temp_r8 = np.zeros(n)
    temp_r4 = np.zeros(n)
    r16_squared = 256
    r8_squared = 64
    r4_squared = 16

    
    sick_list = np.where((S==1)&(isolated !=1))[0]
    xy_array = np.array([[x[i],y[i]] for i in range(n)])

    for sickos in sick_list:
        sick_coords = np.array([x[sickos], y[sickos]])

        list_of_16_hits = np.where(np.sum((xy_array-sick_coords)**2 , axis = 1)<=r16_squared)
        list_of_8_hits = np.where(np.sum((xy_array-sick_coords)**2 , axis = 1)<=r8_squared)
        list_of_4_hits = np.where(np.sum((xy_array-sick_coords)**2 , axis = 1)<=r4_squared)

        temp_r16[list_of_16_hits] +=1
        temp_r8[list_of_8_hits] +=1
        temp_r4[list_of_4_hits] +=1
    
    # It should not count itself as a person in its vacinity, so remove 1 from the sick indexes
    temp_r16[sick_list] -= 1
    temp_r8[sick_list]  -= 1
    temp_r4[sick_list]  -= 1

    R_16[t%10] = temp_r16
    R_8[t%10] = temp_r8
    R_4[t%10] = temp_r4

def initial_testing():
  test_priority = np.argsort(temperatures)
  test_priority = test_priority[-100:-1]
  rand_selected = np.random.randint(0,100,test_capacity)
  to_be_tested = test_priority[rand_selected]
  testing_outcome = np.zeros(test_capacity)
  for agents in to_be_tested: 
    if S[agents] == 1:
      testing_outcome[agents] = 1

    test_results[t*test_capacity : (t+1)*test_capacity] = testing_outcome
  
  index_list[t*test_capacity:(t+1)*test_capacity] = to_be_tested

def gen_information_to_peter():
  agent_to_peter_index = index_list[t*test_capacity:(t+1)*test_capacity]
  start_time = max(0, (t-9)%10)

  CR_tensor = np.zeros(test_capacity,5,10)
  
  for i in range(30):
    CR_tensor[i] = [R_4[start_time:t%10], R_8[start_time:t%10], R_16[start_time:t%10], np.sum(contact_i[start_time:t%10],0), contact_q[start_time:t%10]]
  if t>20:
    information_tensor = np.append(information_tensor, CR_tensor)
  else: information_tensor[t*test_capacity:(t+1)*test_capacity] = CR_tensor

def peter_test(peter_test_list):
  pass

def peter_isolate(peter_isolate_list):
  pass

def man_made_test_agents():
    # Tests sick agents, if positive test then set in isolation and isolate neighbours in contactmatrix
    if t > 20:

        d_type = [('Clist', np.int16), ('Temp', np.float16)]
        test_priority = np.zeros((n,), dtype=d_type)
        test_priority['Clist'] = contact_i[t % 10]
        test_priority['Temp'] = temperatures
        test_priority = np.argsort(test_priority, order=('Clist', 'Temp'))

        i = 0
        tests_made = 0
        while tests_made < test_capacity and i < n - 1:  # can't use more tests than allowed, and can't test more agents than there are agents
            test_person = test_priority[-i - 1]

            if isolated[test_person] != 1:  # Proceed if the selected agent is not already isolated
                tests_made += 1  # A test is counted

                if S[test_person] == 1:  # Isolate sick testsubjects
                    isolated[test_person] = 1

            i = i + 1
        print('Time = ', t,'Tests made: ', tests_made)


def update_states():
    for i in np.where((isolated != 1) & (S == 1) & (np.random.random(n) < B))[0]:  # loop over infecting agents
        temperatures[(x == x[i]) & (y == y[i]) & (S == 0)] = np.random.normal(40,1)  # Raise newly sick agents temperatures
        S[(x == x[i]) & (y == y[i]) & (S == 0)] = 1  # Susceptiples together with infecting agent becomes infected
    for i in np.where((S == 1) & (np.random.random(n) < My))[0]:
        S[i] = 3
    recovered_list = np.where((S == 1) & (np.random.rand(n) < G))[0]
    S[recovered_list] = 2
    # isolated[recovered_list] = 0
    gen_contacts()
    gen_R()


def set_temps():
    for i in np.where(S == 1)[0]:
        temperatures[i] = np.random.normal(37.4, 1.2)

    for i in np.where(temperatures == 0)[0]:
        temperatures[i] = np.random.normal(36.8, 1.0)


if __name__ == '__main__':

    # Parameters of the simulation
    n = 800  # Number of agents
    initial_infected = 10  # Initial infected agents
    N = 1000  # Simulation time
    l = 30  # Lattice size
   
    D_noll = 0.8
    D_reduced = 0.1

    D = D_noll
    B = 0.6
    G = 0.03

    My = 0.00
    start_lock = 50
    lockdown_enabled = False
    test_capacity = 30
    
    t = 0


    #initiate the lists
    x, y, S, isolated, temperatures, tested, nx, ny = __init__()
    set_temps()


    # Contact matrix
    contact_tot = np.zeros((50, n), dtype='int16')
    contact_i = np.zeros((50, n), dtype='int16')
    contact_q = np.zeros((50, n), dtype='float16')
    R_4 = np.zeros((10, n))
    R_8 = np.zeros((10, n))
    R_16 = np.zeros((10, n))

    information_tensor = np.zeros((20*test_capacity, 5, 10))
    test_results = np.zeros((20*test_capacity))

    # output_results = np.zeros(n)

    index_list = np.zeros((150*test_capacity))

    # Plot list

    susceptible_history =  np.zeros(N)
    infected_history = np.zeros(N)
    recovered_history = np.zeros(N)
    dead_history =  np.zeros(N)
    isolation_history = np.zeros(N)


    # Canvas info

    res = 500  # Animation resolution
    tk = Tk()
    tk.geometry(str(int(res * 1.1)) + 'x' + str(int(res * 1.3)))
    tk.configure(background='white')

    canvas = Canvas(tk, bd=2)  # Generate animation window
    tk.attributes('-topmost', 0)
    canvas.place(x=res / 20, y=res / 20, height=res, width=res)
    ccolor = ['#0008FF', '#DB0000', '#12F200', '#68228B', '#000000']

    show_plot = Button(tk, text='Plot', command=plot_sir)
    show_plot.place(relx=0.05, rely=0.85, relheight=0.06, relwidth=0.15)

    

    particles = []
    R = .5  # agent plot radius
    for j in range(n):  # Generate animated particles in Canvas
        particles.append(canvas.create_oval((x[j]) * res / l,
                                            (y[j]) * res / l,
                                            (x[j] + 2 * R) * res / l,
                                            (y[j] + 2 * R) * res / l,
                                            outline=ccolor[0], fill=ccolor[0]))

    # Modifiable parameters by the user

    

    while t < 1000 and list(np.where(S == 1)[0]):
        nx, ny = update_position()
        update_states()

        for j in range(n):
            canvas.move(particles[j], (nx[j] - x[j]) * res / l, (ny[j] - y[j]) * res / l)  # Plot update - Positions
            canvas.itemconfig(particles[j], outline='#303030',
                              fill=ccolor[int(S[j]) if isolated[j] == 0 else 4])  # Plot update - Colors
        tk.update()
        tk.title('Infected:' + str(np.sum(S == 1)) + ' Timesteps passed:' + str(t))
        man_made_test_agents()

        # lockdown_enabled loop
        if start_lock < t < start_lock + 200 and lockdown_enabled:
            D = D_reduced
        else:
            D = D_noll

        x = nx  # Update x
        y = ny  # Update y

        # Used for plotting the graph
        susceptible_history[t] =  len(list(np.where(S == 0)[0]))
        infected_history[t] = len(list(np.where(S == 1)[0]))
        recovered_history[t] = len(list(np.where(S == 2)[0]))
        dead_history[t] =  len(list(np.where(S == 3)[0]))
        isolation_history[t] = len(list(np.where(isolated == 1)[0]))

        t += 1

        if t % 300 == 0:
            plot_sir()

    Tk.mainloop(canvas)  # Release animation handle (close window to finish)

import numpy as np
from tkinter import *
import matplotlib.pyplot as plt
import sys
import copy
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import np_utils #Needed to enable "to_categorical"


np.seterr(invalid='ignore')


def setupNN():
  
    model = Sequential()  # Define the NN model
   
    model.add(Flatten())
    model.add(Dense(50,  activation='relu'))  # Add Layers (Shape kanske inte behövs här?)
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
   
    model.add(Dense(1, activation='sigmoid'))  # softmax ensures number between 0-1.
    model.compile(loss = 'mean_squared_error', optimizer='adam', metrics='accuracy')
    return model


def trainNN():
    reshaped_CR_tensor = np.reshape(CR_tensor, (600,50))
    reshaped_test_results = np.reshape(test_results, 600)
    # Setup the training lists and feed them to the NN
    # Input för NN
    # arry/listan för y_train består av lång lista som korresponderar till x_train där varje index är 0 för frisk eller 1 för sjuk.
    model.fit(reshaped_CR_tensor, reshaped_test_results, epochs=100) #vilken batch size?  #Input för NN, lista, där varje plats är matrix som i artikeln
    # model.evaluate(x_test, y_test, verbose=1
    # model.layers[3].output  # Output för NN, Behöver eventuellt ändra idex beroende på om dropout räknas som lager, vill få output från softmax
    # model.summary() Få tag i info om modellens

   
def make_predictionsNN():
   
    slicing_list = [(t-j)%10 for j in range(10) ]
    for i in range(n):
        n_tensor[i] = np.array([R_4[slicing_list, i], R_8[slicing_list, i], R_16[slicing_list, i],
        total_contact_i[slicing_list, i], contact_q[slicing_list, i]])
 
    resultNN = model.predict(np.reshape(n_tensor, (n, 50)))
    return resultNN
    # agent_to_peter_index = index_list[t*test_capacity:(t+1)*test_capacity]
 
    #Tensor for prediction regarding all agents 
   
 
def deployNN():
    resultNN = make_predictionsNN()
    most_plausibly_sick_agents  = np.where(resultNN>0.995)[0]
    peter_isolate(most_plausibly_sick_agents)
 
    maybe_sick_agents = np.where((0.5<resultNN) & (resultNN<=0.995))[0]
    rising_probability_indexes = np.argsort(maybe_sick_agents)
    if len(list(rising_probability_indexes))>30:
        returned_results = peter_test(rising_probability_indexes[-31:-1])
    else:
        returned_results = peter_test(rising_probability_indexes)
    # Gör något med returnerade resultaten också
 
def init_mult():
    x = np.zeros(n)
    y = np.zeros(n)
    S = np.zeros(n)  # status array, 0: Susceptiple, 1: Infected, 2: Recovered, 3: Dead
 
    for i in range(num_of_cities):
        S[i*n//num_of_cities: (i*n//num_of_cities + initial_infected//num_of_cities)] = 1
        citizenship[i*n//num_of_cities: (i*n//num_of_cities + initial_infected//num_of_cities)] = i
    x = np.floor(np.random.rand(n) * l/2)  
    y = np.floor(np.random.rand(n) * l)
    x_init = copy.deepcopy(x)
    y_init = copy.deepcopy(y)
    isolated = np.zeros(n)  # Isolation array, 0: not isolated, 1: Is currently in isolation
    temperatures = np.zeros(n, dtype='float16')  # temperature array
    tested = np.zeros(n)
    nx = x  # updated x
    ny = y  # updated y
    return x_init, y_init, x, y, S, isolated, temperatures, tested, nx, ny

def __init__():
    if num_of_cities>1:
        x_init, y_init, x, y, S, isolated, temperatures, tested, nx, ny = init_mult()
        return x_init, y_init, x, y, S, isolated, temperatures, tested, nx, ny
    x = np.floor(np.random.rand(n) * l)  # x coordinates
    y = np.floor(np.random.rand(n) * l)  # y coordinates
    x_init = copy.deepcopy(x)
    y_init = copy.deepcopy(y)
    S = np.zeros(n)  # status array, 0: Susceptiple, 1: Infected, 2: recovered, 3: Dead
    isolated = np.zeros(n)  # Isolation array, 0: not isolated, 1: Is currently in isolation
    temperatures = np.zeros(n, dtype='float16')  # temperature array
    tested = np.zeros(n)
    S[0:initial_infected] = 1  # Infect random agents
    nx = copy.deepcopy(x)  # updated x
    ny = copy.deepcopy(y)  # updated y
    setupNN()
    return x_init, y_init, x, y, S, isolated, temperatures, tested, nx, ny
 
 
# Plots graph
def plot_sir():
    index_list_for_plot = np.array([i for i in range(t)])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    label_susceptible = 'Susceptible = ' + str(susceptible_history[t-1])
    label_recovered = 'Recovered = ' + str(recovered_history[t-1])
    label_infected = 'Infected = ' + str(infected_history[t-1])
    label_dead = 'Dead = ' + str(dead_history[t-1])
    label_isolation = 'Isolation = ' + str(isolation_history[t-1])
    ax.plot(index_list_for_plot, susceptible_history[:t], color='blue', label=label_susceptible)
    ax.plot(index_list_for_plot, recovered_history[:t], color='green', label=label_recovered)
    ax.plot(index_list_for_plot, infected_history[:t], color='red', label=label_infected)
    ax.plot(index_list_for_plot, dead_history[:t], color='purple', label=label_dead)
    ax.plot(index_list_for_plot, isolation_history[:t], color='black', label=label_isolation)
    ax.set_title('Infection plot')
    ax.legend()
    plt.show()


def update_position():
    k = 0.04
    for agent in range(n):
        prob_x = [max(0,1/3 +k*(x[agent]-x_init[agent])), 1/3, max(0, 1/3-k*(x[agent]-x_init[agent]))]
        prob_x /= sum(prob_x)
        prob_y = [max(0, 1/3 +k*(y[agent]-y_init[agent])), 1/3, max(0, 1/3-k*(y[agent]-y_init[agent]))]
        prob_y /= sum(prob_y)
        dx = np.random.choice([-1, 0, 1], p=np.array(prob_x))
        dy = np.random.choice([-1, 0, 1], p=np.array(prob_y))
        nx[agent] += dx
        ny[agent] += dy
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
 
    total_contact_i[t%10] = np.sum(contact_i, 0)
    total_contact_tot[t%10] = np.sum(contact_tot, 0)
 
    contact_q[t % 10] =  np.nan_to_num(np.divide(np.sum(contact_i, 0),np.sum(contact_tot, 0)))
 
 
def gen_R():  # Generatorfunktion för R-matriserna
    if num_of_cities > 1 and False:
        mult_gen_R()
    else:
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

def mult_gen_R():
    temp_r16 = np.zeros(n)
    temp_r8 = np.zeros(n)
    temp_r4 = np.zeros(n)
    r16_squared = 256
    r8_squared = 64
    r4_squared = 16

    for i in range(num_of_cities):
            
        sick_list = np.where((S==1)&(isolated !=1)&(citizenship == i))[0]
        xy_array = np.array([[x[i],y[i]] for i in np.where(citizenship == i)])

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
    rand_selected = np.random.randint(0,99,test_capacity)
    to_be_tested = test_priority[rand_selected]
    testing_outcome = np.zeros(test_capacity)
    for agents in range(30):
        if S[to_be_tested[agents]] == 1:
            testing_outcome[agents] = 1
            isolated[to_be_tested[agents]] = 1
 
    test_results[t] = testing_outcome
    index_list[t*test_capacity:(t+1)*test_capacity] = to_be_tested
 
    gen_information_to_peter(to_be_tested)
 
 
def gen_information_to_peter(to_be_tested):
    # agent_to_peter_index = index_list[t*test_capacity:(t+1)*test_capacity]
 
    #Tensor for prediction regarding all agents
    slicing_list = [(t-j)%10 for j in range(10) ]
   
 
    for i in range(test_capacity):
        k = to_be_tested[i]
        CR_tensor[t][i] = np.array([R_4[slicing_list, k] , R_8[slicing_list, k], R_16[slicing_list, k],
        total_contact_i[slicing_list, k], contact_q[slicing_list, k]])
   
    information_tensor[t*test_capacity:(t+1)*test_capacity] = CR_tensor[t]
 
 
def peter_test(peter_test_list):
   
    results_from_peters_test = np.zeros(test_capacity)
    i = 0
   
    for agent in peter_test_list:
        if S[agent] == 1:
            results_from_peters_test[i] = 1
            isolated[agent] = 1
        i +=1
   
    return results_from_peters_test
 
def peter_isolate(peter_isolate_list):
 
    for agent in peter_isolate_list:
        isolated[agent] = 1
 
 
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
 
 
def update_states():
    for i in np.where((isolated != 1) & (S == 1) & (np.random.random(n) < B))[0]:  # loop over infecting agents
        temperatures[(x == x[i]) & (y == y[i]) & (S == 0)] = np.random.normal(40,1)  # Raise newly sick agents temperatures
        S[(x == x[i]) & (y == y[i]) & (S == 0)] = 1  # Susceptiples together with infecting agent becomes infected
    for i in np.where((S == 1) & (np.random.random(n) < My))[0]:
        S[i] = 3
    recovered_list = np.where((S == 1) & (np.random.rand(n) < G))[0]
    S[recovered_list] = 2
    isolated[recovered_list] = 0
    gen_contacts()
    gen_R()
 
 
def set_temps():
    for i in np.where(S == 1)[0]:
        temperatures[i] = np.random.normal(37.4, 1.2)
 
    for i in np.where(S == 0)[0]:
        temperatures[i] = np.random.normal(36.8, 1.0)
 
if __name__ == '__main__':
 
    # Currently fixed parameters of the simulation
    n = 800  # Number of agents
    initial_infected = 40  # Initial infected agents
    N = 1000  # Simulation time
    l = 30  # Lattice size
    num_of_cities = 2
    citizenship = np.zeros(n)
 
    #initiate the lists
    x_init, y_init, x, y, S, isolated, temperatures, tested, nx, ny = __init__()
    set_temps()
 
    # All things related to the GUI
 
    geores = 500  # Animation resolution
    res = geores*1.4
    tk = Tk()
    tk.geometry(str(int(geores * 1.5)) + 'x' + str(int(geores * 1.9)))
    tk.configure(background='white')
 
    canvas = Canvas(tk, bd=2)  # Generate animation window
    tk.attributes('-topmost', 0)
    canvas.place(x=res / 20, y=res / 20, height=res, width=res)
    ccolor = ['#0008FF', '#DB0000', '#12F200', '#68228B', '#000000']
   
    def restart():
        num_of_cities = num_of_cities_selector
 
        # Consider if n and l should be modifieable
        # n = num_of_agents.get()
        # l = lattice_size.get()
        S = np.zeros(n)
        for i in range(num_of_cities):
            S[i//num_of_cities[-1]:i//num_of_cities[-1] + initial_infected//num_of_cities[-1]]
       
    show_plot = Button(tk, text='Plot', command=plot_sir)
    show_plot.place(relx=0.05, rely=0.85, relheight=0.04, relwidth=0.15)
 
    rest = Button(tk, text='Restart',command= restart)
    rest.place(relx=0.05, rely=.9, relheight= 0.04, relwidth= 0.15 )
 
    Beta = Scale(tk, from_=0, to=1, orient=HORIZONTAL, label='Infection probability', font=("Helvetica", 8),resolution=0.01)
    Beta.place(relx=.22, rely=.85, relheight= 0.08, relwidth= 0.23)    
    Beta.set(0.8)            # Parameter slider for infection rate                                                      
 
    Gamma = Scale(tk, from_=0, to=0.1, orient=HORIZONTAL, label='Recovery rate', font=("Helvetica", 8) ,resolution=0.001)
    Gamma.place(relx=.47, rely=.85, relheight= 0.08, relwidth= 0.23)
    Gamma.set(0.01)          # Parameter slider for recovery rate
 
    Diff = Scale(tk, from_=0, to=1, orient=HORIZONTAL, label='Diffusion probability', font=("Helvetica", 8),resolution=0.01)
    Diff.place(relx=.72, rely=.85, relheight= 0.08, relwidth= 0.23)
    Diff.set(0.5)            # Parameter slider for diffusion rate
 
    Mortality = Scale(tk, from_=0, to=1, orient=HORIZONTAL, label='Mortality probability', font=("Helvetica", 8),resolution=0.01)
    Mortality.place(relx=.72, rely=.93, relheight= 0.08, relwidth= 0.23)
    Mortality.set(0.01)            # Parameter slider for Mortality
 
    # Num of Socs selection
    num_of_cities_selector = 0         # Creating a variable which will track the selected checkbutton
    buttonlist = []                    # Empty list which is going to hold all the checkbuttons
    available_num_of_societies = [1,2,4,6,8]                
    for i in range(5):
        buttonlist.append(Checkbutton(tk,text = i ,onvalue = available_num_of_societies[i], variable = num_of_cities_selector))  
                          #Creating and adding checkbutton to list
        buttonlist[i].place(relx =0.23 + 0.03*i, rely = .96 )
 
    particles = []
    R = .5  # agent plot radius
    x_offset = np.zeros(n)
    y_offset = np.zeros(n)
    for i in range(n):
        if i < n//2:
            x_offset[i] = 0
        else: x_offset[i] = l/2 +3
    for j in range(n):  # Generate animated particles in Canvas
            particles.append(canvas.create_oval((x[j]+ x_offset[j]) * res / l,
                                                (y[j] +  y_offset[j]) * res / l,
                                                (x[j] +  x_offset[j] + 2 * R) * res / l,
                                                (y[j] +  y_offset[j]+ 2 * R) * res / l,
                                                outline=ccolor[0], fill=ccolor[0]))
 
    D_noll = Diff.get()
    D_reduced = 0.1
    D = D_noll
    B = Beta.get()
    G = Gamma.get()
    My = Mortality.get()
    start_lock = 50
    lockdown_enabled = False
    test_capacity = 30
    t = 0
    peter_start_time = 20
 
    # Contact matrix
    contact_tot = np.zeros((50, n), dtype='int16')
    contact_i = np.zeros((50, n), dtype='int16')
    total_contact_tot = np.zeros((10, n), dtype='int16')
    total_contact_i = np.zeros((10, n), dtype='int16')
    contact_q = np.zeros((50, n), dtype='float16')
    R_4 = np.zeros((10, n))
    R_8 = np.zeros((10, n))
    R_16 = np.zeros((10, n))
    CR_tensor = np.zeros((peter_start_time, test_capacity,5,10))
    n_tensor = np.zeros((n,5,10))
    information_tensor = np.zeros((20*test_capacity, 5, 10))
    test_results = np.zeros((20,test_capacity))
 
    # output_results = np.zeros(n)
    index_list = np.zeros((150*test_capacity))
 
    # Plot lists
    susceptible_history =  np.zeros(N)
    infected_history = np.zeros(N)
    recovered_history = np.zeros(N)
    dead_history =  np.zeros(N)
    isolation_history = np.zeros(N)    
    model = setupNN()
 
    while t < 1000:
 
        # if dynamic_update: ... ? Vi kan lägga till så att endast om man har klickat i att man vill
        # kunna ändra params under körning så uppdateras de
        nx, ny = update_position()
        update_states()
       
        if t<20:
            initial_testing()
        if t == 20:
            trainNN()      
        if t>20:
            deployNN()
 
        for j in range(n):
            canvas.move(particles[j], (nx[j] - x[j]) * res / l, (ny[j] - (y[j])) * res / l)  # Plot update - Positions
            canvas.itemconfig(particles[j], outline='#303030',
                              fill=ccolor[int(S[j]) if isolated[j] == 0 else 4])  # Plot update - Colors
        tk.update()
        tk.title('Infected:' + str(np.sum(S == 1)) + ' Timesteps passed:' + str(t))
 
        # lockdown_enabled loop
        if start_lock < t < start_lock + 200 and lockdown_enabled:
            D = D_reduced
        else:
            D = D_noll
 
        x = copy.deepcopy(nx)  # Update x
        y = copy.deepcopy(ny)  # Update y
 
        # Used for plotting the graph
        susceptible_history[t] =  len(list(np.where(S == 0)[0]))
        infected_history[t] = len(list(np.where(S == 1)[0]))
        recovered_history[t] = len(list(np.where(S == 2)[0]))
        dead_history[t] =  len(list(np.where(S == 3)[0]))
        isolation_history[t] = len(list(np.where(isolated == 1)[0]))
        t += 1
 
        if t % 90 == 0:
            plot_sir()
 
    Tk.mainloop(canvas)
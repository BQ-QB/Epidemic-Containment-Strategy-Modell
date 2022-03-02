import numpy as np
import matplotlib.pyplot as plt

def __init__():
    x = np.floor(np.random.rand(n) * l)  # x coordinates
    y = np.floor(np.random.rand(n) * l)  # y coordinates
    S = np.zeros(n)  # status array, 0: Susceptiple, 1: Infected, 2: recovered, 3: Dead
    isolated = np.zeros(n)  # Isolation array, 0: not isolated, 1: Is currently in isolation
    temperatures = np.zeros(n, dtype='float16')  # temperature array
    tested = np.zeros(n)
    aPosition = zip(x,y)
    S[0:initial_infected] = 1  # Infect agents that are close to center
    nx = x  # updated x
    ny = y  # updated y
    print(aPosition)


    return x, y, S, isolated, temperatures, tested, nx, ny

def plot_canvas(t):
    plt.plot()
    pass


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
    coord_list = np.zeros(n)
    contact_list = np.zeros(n)
    sick_contact_list = np.zeros(n)

    for agent in range(n):
        coord_list[agent] = (2 ** x[agent]) * (3 ** y[agent])

    for infected in np.argsort(np.where((S == 1) & (isolated != 1))[0]):
        infected_agent = infected
        for other_agent in np.where(((S == 1) | (S == 0)) & (isolated == 0))[0]:
            if (coord_list[infected_agent] == coord_list[other_agent]) & (infected_agent != other_agent):
                sick_contact_list[other_agent] += 1

    for agent in range(n):
        current_agent = agent
        for another_agent in range(n):
            if (coord_list[current_agent] == coord_list[another_agent]) & (another_agent != current_agent):
                contact_list[current_agent] += 1

    contact_i[t % 50] = sick_contact_list
    contact_tot[t % 50] = contact_list

    total_contact_tot[t % 10] = np.sum(contact_tot, 0)
    total_contact_i[t % 10] = np.sum(contact_i, 0)

    contact_q[t % 10] = np.nan_to_num(np.divide(total_contact_i[t % 10], total_contact_tot[t % 10]))


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
        temperatures[(x == x[i]) & (y == y[i]) & (S == 0)] = np.random.normal(40,
                                                                              1)  # Raise newly sick agents temperatures
        S[(x == x[i]) & (y == y[i]) & (S == 0)] = 1  # Susceptiples together with infecting agent becomes infected
    for i in np.where((S == 1) & (np.random.random(n) < My))[0]:
        S[i] = 3
    recovered_list = np.where((S == 1) & (np.random.rand(n) < G))[0]
    S[recovered_list] = 2
    # isolated[recovered_list] = 0
    gen_contacts()
  


def set_temps():
    for i in np.where(S == 1)[0]:
        temperatures[i] = np.random.normal(37.4, 1.2)

    for i in np.where(temperatures == 0)[0]:
        temperatures[i] = np.random.normal(36.8, 1.0)


if __name__ == '__main__':
   
    # Parameters of the simulation
    n = 800  # Number of agents
    initial_infected = 10  # Initial infected agents
    N = 100000  # Simulation time
    l = 30  # Lattice size

    # Modifiable parameters by the user

    D_noll = 0.8
    D_reduced = 0.1

    D = D_noll
    B = 0.6
    G = 0.03

    My = 0.00
    start_lock = 50
    lockdown_enabled = False
    test_capacity = 30
    x, y, S, isolated, temperatures, tested, nx, ny = __init__()

    set_temps()
    t = 0


    # Historylists used for plotting SIR-graph
    infected_history = np.array([initial_infected - 1])
    susceptible_history = np.array([n - initial_infected + 1])
    recovered_history = np.array([0])
    dead_history = np.array([0])
    isolation_history = np.array([0])
   

    # Contact matrix
    contact_tot = np.zeros((50, n), dtype='int16')
    contact_i = np.zeros((50, n), dtype='int16')
    contact_q = np.zeros((50, n), dtype='float16')
    total_contact_i = np.zeros((10, n), dtype='int16')
    total_contact_tot = np.zeros((10, n), dtype='int16')
    R_4 = np.zeros((10, n))
    R_8 = np.zeros((10, n))
    R_16 = np.zeros((10, n))

    information_tensor = np.zeros((20*test_capacity, 5, 10))
    test_results = np.zeros((20*test_capacity))

    # output_results = np.zeros(n)

    index_list = np.zeros((150*test_capacity))
       
    

    while t < 1000 and list(np.where(S == 1)[0]):
        nx, ny = update_position()
        update_states()
        man_made_test_agents()
       

        


        
        # lockdown_enabled loop
        if start_lock < t < start_lock + 200 and lockdown_enabled:
            D = D_reduced
        else:
            D = D_noll

        x = nx  # Update x
        y = ny  # Update y

        # Used for plotting the graph
        susceptible_history = np.append(susceptible_history, len(list(np.where(S == 0)[0])))
        infected_history = np.append(infected_history, len(list(np.where(S == 1)[0])))
        recovered_history = np.append(recovered_history, len(list(np.where(S == 2)[0])))
        dead_history = np.append(dead_history, len(list(np.where(S == 3)[0])))
        isolation_history = np.append(isolation_history, len(list(np.where(isolated == 1)[0])))

        t += 1

        if t % 10 == 0:
            plot_sir()
    
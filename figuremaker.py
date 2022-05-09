from cProfile import label
from re import I
from matplotlib import figure
import numpy as np
import matplotlib.pyplot as plt

def better_fig(filename):    
    """ From the results of one or many simulations, displays a plot of the average (solid lines) and the 95% confidence interval

    Args:
        result_tensor : Loaded data to create plot from
        load_bool : If the data is not present, but needs to be loaded in
        file_path : Where potential data should be loaded from
        save_plot: If the created plot should be saved

    """
 
    data_tensor = np.load(filename) 
   

    simulation_time = len(data_tensor[0][0])
    number_of_simulations = len(data_tensor)     
   
    tensor_sus = np.zeros((simulation_time,number_of_simulations))
    tensor_inf = np.zeros((simulation_time,number_of_simulations))
    tensor_rec = np.zeros((simulation_time,number_of_simulations))
    tensor_dead = np.zeros((simulation_time,number_of_simulations))
    tensor_iso = np.zeros((simulation_time, number_of_simulations))
    low_sus = np.zeros(simulation_time)
    low_inf = np.zeros(simulation_time)
    low_rec = np.zeros(simulation_time)
    low_dead = np.zeros(simulation_time)
    low_iso = np.zeros(simulation_time)
    high_sus = np.zeros(simulation_time)
    high_inf = np.zeros(simulation_time)
    high_rec = np.zeros(simulation_time)
    high_dead = np.zeros(simulation_time)
    high_iso = np.zeros(simulation_time)
    std_sus = np.zeros(simulation_time)
    std_inf = np.zeros(simulation_time)
    std_rec = np.zeros(simulation_time)
    std_dead = np.zeros(simulation_time)
    std_iso = np.zeros(simulation_time)
    

    # Calculate the mean of the different states
    mean_res = np.mean(data_tensor, axis = 0)
    z = 1.96
    sn = np.sqrt(number_of_simulations)
    
    # Divide the data from a timestep into different tensors for the different states, then create the values to plot
    for i in range(simulation_time):
        tensor_sus[i] = [item[0][i] for item in data_tensor]
        tensor_inf[i] = [item[1][i] for item in data_tensor]
        tensor_rec[i] = [item[2][i] for item in data_tensor]
        tensor_dead[i] = [item[3][i] for item in data_tensor]
        tensor_iso[i] = [item[4][i] for item in data_tensor]
       
        

        std_sus[i] = np.std(tensor_sus[i])
        low_sus[i] = np.maximum(0.,mean_res[0][i] - z/sn*std_sus[i])
        high_sus[i] = mean_res[0][i] + z/sn*std_sus[i]

        std_inf[i] = np.std(tensor_inf[i])
        low_inf[i] = np.maximum(0.,mean_res[1][i] - z/sn*std_inf[i])
        high_inf[i] = mean_res[1][i] + z/sn*std_inf[i]

        std_rec[i] = np.std(tensor_rec[i])
        low_rec[i] = np.maximum(0.,mean_res[2][i] - z/sn*std_rec[i])
        high_rec[i] = mean_res[2][i] + z/sn*std_rec[i]

        std_dead[i] = np.std(tensor_dead[i])
        low_dead[i] = np.maximum(0.,mean_res[3][i] - z/sn*std_dead[i])
        high_dead[i] = mean_res[3][i] + z/sn*std_dead[i]

        std_iso[i] = np.std(tensor_iso[i])
        low_iso[i] = np.maximum(0.,mean_res[4][i] - z/sn*std_iso[i])
        high_iso[i] = mean_res[4][i] + z/sn*std_iso[i]
   

    # Plotting
    fig, ax = plt.subplots(1, 1, figsize = (6,6))
    ax.plot(mean_res[0], c = 'b', label = 'Mottagliga')
    x = list(range(len(low_sus)))
    
    ax.fill_between(x, low_sus, high_sus, color = '#ADD8E6')
    ax.plot(mean_res[1], c = 'r', label = 'Sjuka')
    ax.fill_between(x, low_inf, high_inf, color = '#FF7F7F')
    ax.plot(mean_res[2], c = 'g', label = 'Återhämtade')
    ax.fill_between(x, low_rec, high_rec, color = '#90EE90')
    ax.plot(mean_res[3], c = '#AE08FB', label = 'Döda') #plot death
    ax.fill_between(x, low_dead, high_dead, color = '#CBC3E3')
    ax.plot(mean_res[4], c = 'k', label = 'Isolerade' ) 
    ax.fill_between(x, low_iso, high_iso, color = '#D3D3D3')
    
    
    ax.set(xlim=(-1, 151), ylim=(-5, 10050))
    ax.set_ylabel('Antal agenter')
    ax.set_xlabel('Tid')
    ax.set_title('')
    ax.axvline( x= 20, color = 'k', linestyle = 'dotted', label = 'Bekämpningsstrategi startar')
    #ax.axvline( x= 50, color = 'y', linestyle = 'dotted', label = 'Mutation inträffar')
    ax.set_xticks([0, 25, 50, 75, 100, 125, 149])
    ax.set_xticklabels([0, 25, 50, 75, 100, 125, 150])
    #ax.legend(fancybox=True, framealpha=0.8, loc = 1)
    plt.show()
    

better_fig('./SIR_peter_regular.npy')
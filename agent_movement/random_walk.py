import numpy as np

# summary: Self containted function, documented with docstrings and comments.
# further improvments: vectorize operations with masked arrays (isolated_agents, susceptible_agents). In that case, an example would be "not isolated" instead of "isolated != 1"!
# suggestion: Implement indiviual k for each agent, redfine x_position, y_position into single variable, define agent state constant e.g. DEAD = 3 (one acceptable use of global variable due to contant value)

def random_walk(x_position, y_position, x_initial, y_inititial, isolated_agents, susceptible_agents): # random_walk is the new update_position()
    """This function moves every agent in a random direction (which also can be no movement), either up, down, left or right.

    Args:
        x_position (numpy.nd.array): agent's x coordinate. (previously x)
        y_position (numpy.nd.array): agent's y coordinate. (previously y)
        x_initial (numpy.nd.array): agent's starting x coordinate (previously x_init)
        y_inititial (numpy.nd.array): agent's starting y coordinate (previously y_init)
        isolated_agents (numpy.nd.array):  mask for isolated agents   
        susceptible_agents (numpy.nd.array): mask for susceptible_agents

    Returns:
        x_position (numpy.nd.array): agent's x coordinate after the random walk
        y_position (numpy.nd.array): agent's y coordinate after the random walk
    """
    k = 0.04    # a factor that determines max walking distance from start position. Maximum of movement is dfined by k * max_step = 1/3
    for agent in range(len(x)): # It can even be len(y), due to len(x) == len (y) == number of agents
        prob_x = [
            max(0, 1/3 + k*(x_position[agent]-x_initial[agent])),   # probability to go left
            1/3,    # probability to stay
            max(0, 1/3 - k*(x_position[agent]-x_initial[agent])) # probability to go right
        ]

        prob_x /= sum(prob_x)   # normalises the probablities

        prob_y = [
            max(0, 1/3 + k*(y_position[agent]-y_inititial[agent])),
            1/3,
            max(0, 1/3 - k*(y_position[agent]-y_inititial[agent]))
        ]

        prob_y /= sum(prob_y)

        dx = np.random.choice([-1, 0, 1], p=np.array(prob_x))   # makes a random choice of one of 3 main movements for the agent's x coordiante
        dy = np.random.choice([-1, 0, 1], p=np.array(prob_y))   # makes a random choice of one of 3 main movements for the agent's y coordiante

        x_position[agent] += dx if not (isolated_agents[agent] == 1 or susceptible_agents[agent] == 3) else 0
        # makes the random movement for the agentx coordinate happen, unless the agent is either isolated or dead

        y_position[agent] += dy if not (isolated_agents[agent] == 1 or susceptible_agents[agent] == 3) else 0
        # makes the random movement for the agent's y coordinate happen, unless the agent is either isolated or dead

    return x_position, y_position   # outputs the agent's x and y coordinate after the (supposed) random walk!
import numpy as np


def get_image(agent_loc, goal_loc):
    agent_loc = (int((2-agent_loc[1])*100), int((2 + agent_loc[0])*100))
    goal_loc = (int((2-goal_loc[1])*100), int((2 + goal_loc[0])*100))
    # create bg
    img = np.ones((400, 400, 3), dtype=np.uint8)*255

    # create vertical and horizontal boundary
    img[199:201, :, :] = 0
    img[:, 199:201, :] = 0

    # indicate agent
    a_x = agent_loc[0]
    a_y = agent_loc[1]
    img[a_x-3:a_x+3, a_y-3:a_y+3] = [255, 0, 0]

    # indicate goal
    g_x = goal_loc[0]
    g_y = goal_loc[1]
    img[g_x-3:g_x+3, g_y-3:g_y+3] = [0, 255, 0]

    return img


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.imshow(get_image((-0.1, 0.1), (0.1, 0.1)))
    plt.show()

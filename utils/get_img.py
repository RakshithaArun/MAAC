import numpy as np
import cv2

def get_image(agent_loc, goal_loc):
    # scale the input coordinates according to size of output
    agent_loc = (int((4 - agent_loc[1])*100), int((4 + agent_loc[0])*100))
    goal_loc = (int((4 - goal_loc[1])*100), int((4 + goal_loc[0])*100))

    # read in images used to represent agent and bomb
    image=cv2.imread('C:\\Users\\HP\\Desktop\\NTU\\FYP\\FYP Code\\MAAC\\utils\\spy.png')
    ag = cv2.resize(image, dsize=(40, 40), interpolation=cv2.INTER_CUBIC)
    image=cv2.imread('C:\\Users\\HP\\Desktop\\NTU\\FYP\\FYP Code\\MAAC\\utils\\bomb.jpg')
    gl=cv2.resize(image, dsize=(40, 40), interpolation=cv2.INTER_CUBIC)

    # create bg
    img = np.ones((800, 800, 3), dtype=np.uint8)*255

    # create vertical and horizontal boundary
    img[399:401, :, :] = 0
    img[:, 399:401, :] = 0

    # indicate agent
    a_x = agent_loc[0]
    a_y = agent_loc[1]
    #img[a_x-3:a_x+3, a_y-3:a_y+3] = [255, 0, 0]
    img[a_x-20:a_x+20, a_y-20:a_y+20] = ag

    # indicate goal
    g_x = goal_loc[0]
    g_y = goal_loc[1]
    img[g_x-20:g_x+20, g_y-20:g_y+20] = gl

    #return (img)
    return np.array([img])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.imshow(get_image((-0.4, 0.5), (0.3, 0.9)))
    plt.show()

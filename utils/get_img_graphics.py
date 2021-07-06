import numpy as np
import cv2

def get_image(agent_loc, goal_loc, ghosts_locs=[]):
    # scale the input coordinates according to size of output
    agent_loc = (int((2 - agent_loc[1])*100), int((2 + agent_loc[0])*100))
    goal_loc = (int((2 - goal_loc[1])*100), int((2 + goal_loc[0])*100))
    
    for ind, location in enumerate(ghosts_locs):
        ghosts_locs[ind] = [int((2 - location[1])*100), int((2 + location[0])*100)]
    # read in images used to represent agent and bomb
    # image=cv2.imread('C:\\Users\\HP\\Desktop\\NTU\\FYP\\FYP Code\\MAAC\\utils\\spy.png')
    # ag = cv2.resize(image, dsize=(40, 40), interpolation=cv2.INTER_CUBIC)
    # image=cv2.imread('C:\\Users\\HP\\Desktop\\NTU\\FYP\\FYP Code\\MAAC\\utils\\bomb.jpg')
    # gl=cv2.resize(image, dsize=(40, 40), interpolation=cv2.INTER_CUBIC)

    # create bg
    img = np.ones((400, 400, 3), dtype=np.uint8)*0
    try:
        # create vertical and horizontal boundary
        img[199:201, :, :] = 255
        img[:, 199:201, :] = 255

        # indicate agent
        a_x = agent_loc[0]
        a_y = agent_loc[1]
        #img[a_x-3:a_x+3, a_y-3:a_y+3] = [255, 0, 0]
        img[a_x-3:a_x+3, a_y-3:a_y+3] = [0,0,255]

        # indicate goal
        g_x = goal_loc[0]
        g_y = goal_loc[1]
        img[g_x-3:g_x+3, g_y-3:g_y+3] = [0,255,0]


        for ghost_loc in ghosts_locs:
            
            gh_x = ghost_loc[0]
            gh_y = ghost_loc[1]

            r = 10
            r2 = 8
            x = np.arange(0, 400)
            y = np.arange(0, 400)
            mask = (x[np.newaxis,:]-gh_y)**2 + (y[:,np.newaxis]-gh_x)**2 < r**2
            img[mask] = [220,220,220]

            img[gh_x-3:gh_x+3, gh_y-3:gh_y+3] = [255,0,0]

        # circle = (g_x - 100) ** 2 + (g_y - 100) ** 2
        # donut = np.logical_and(circle < (6400 + 60), circle > (6400 - 60))
        # m1 = ((g_y-200)**2 + (g_x-100)**2 < 30**2)
        # m2 = ((g_y-350)**2 + (g_x-400)**2 < 20**2)
        # m3 = ((g_y-260)**2 + (g_x-200)**2 < 20**2)
        # img[m1+m2+m3]=1

        # radius = 100
        # cx, cy = 1, 1 # The center of circle
        # y, x = np.ogrid[-radius: radius, -radius: radius]
        # index = x**2 + y**2 <= radius**2
        # img[cy-radius:cy+radius, cx-radius:cx+radius][index] = [0,255,0]
        
        # x = np.linspace(-2, 2, 800)
        # y = np.linspace(-2, 2, 800)
        # mask = np.sqrt((x-g_x)**2+(y-g_y)**2)

    except:
        pass
    #return (img)
    return np.array([img])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.imshow(get_image((-0.4, 0.5), (0.3, 0.9),[[0.1,0.9],[0.2,0.8]])[0])
    plt.show()




    import numpy as np
import cv2

def get_image(agent_loc, goal_loc, ghosts_locs=[]):
    # scale the input coordinates according to size of output
    agent_loc = (int((4 - agent_loc[1])*100), int((4 + agent_loc[0])*100))
    goal_loc = (int((4 - goal_loc[1])*100), int((4 + goal_loc[0])*100))
    
    for ind, location in enumerate(ghosts_locs):
        ghosts_locs[ind] = [int((4 - location[1])*100), int((4 + location[0])*100)]
    # read in images used to represent agent and bomb
    # image=cv2.imread('C:\\Users\\HP\\Desktop\\NTU\\FYP\\FYP Code\\MAAC\\utils\\spy.png')
    # ag = cv2.resize(image, dsize=(40, 40), interpolation=cv2.INTER_CUBIC)
    # image=cv2.imread('C:\\Users\\HP\\Desktop\\NTU\\FYP\\FYP Code\\MAAC\\utils\\bomb.jpg')
    # gl=cv2.resize(image, dsize=(40, 40), interpolation=cv2.INTER_CUBIC)

    # create bg
    img = np.ones((800, 800, 3), dtype=np.uint8)*0
    try:
        # create vertical and horizontal boundary
        img[399:401, :, :] = 255
        img[:, 399:401, :] = 255

        # indicate agent
        a_x = agent_loc[0]
        a_y = agent_loc[1]
        #img[a_x-3:a_x+3, a_y-3:a_y+3] = [255, 0, 0]
        img[a_x-3:a_x+3, a_y-3:a_y+3] = [0,0,255]


        # indicate goal
        g_x = goal_loc[0]
        g_y = goal_loc[1]
        img[g_x-3:g_x+3, g_y-3:g_y+3] = [0,255,0]


        for ghost_loc in ghosts_locs:
            
            gh_x = ghost_loc[0]
            gh_y = ghost_loc[1]

            r = 10
            r2 = 8
            x = np.arange(0, 800)
            y = np.arange(0, 800)
            mask = (x[np.newaxis,:]-gh_y)**2 + (y[:,np.newaxis]-gh_x)**2 < r**2
            img[mask] = [220,220,220]

            img[gh_x-3:gh_x+3, gh_y-3:gh_y+3] = [255,0,0]

        # circle = (g_x - 100) ** 2 + (g_y - 100) ** 2
        # donut = np.logical_and(circle < (6400 + 60), circle > (6400 - 60))
        # m1 = ((g_y-200)**2 + (g_x-100)**2 < 30**2)
        # m2 = ((g_y-350)**2 + (g_x-400)**2 < 20**2)
        # m3 = ((g_y-260)**2 + (g_x-200)**2 < 20**2)
        # img[m1+m2+m3]=1

        # radius = 100
        # cx, cy = 1, 1 # The center of circle
        # y, x = np.ogrid[-radius: radius, -radius: radius]
        # index = x**2 + y**2 <= radius**2
        # img[cy-radius:cy+radius, cx-radius:cx+radius][index] = [0,255,0]
        
        # x = np.linspace(-2, 2, 800)
        # y = np.linspace(-2, 2, 800)
        # mask = np.sqrt((x-g_x)**2+(y-g_y)**2)

    except:
        pass
    #return (img)
    return np.array([img])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.imshow(get_image((-0.4, 0.5), (0.3, 0.9),[[0.1,0.9],[0.2,0.8]])[0])
    plt.show()






    import numpy as np
import cv2

def get_image(agent_loc, goal_loc, ghosts_locs=[]):
    # scale the input coordinates according to size of output
    agent_loc = (int((2 - agent_loc[1])*100), int((2 + agent_loc[0])*100))
    goal_loc = (int((2 - goal_loc[1])*100), int((2 + goal_loc[0])*100))
    
    for ind, location in enumerate(ghosts_locs):
        ghosts_locs[ind] = [int((2 - location[1])*100), int((2 + location[0])*100)]
    # read in images used to represent agent and bomb
    # image=cv2.imread('C:\\Users\\HP\\Desktop\\NTU\\FYP\\FYP Code\\MAAC\\utils\\spy.png')
    # ag = cv2.resize(image, dsize=(40, 40), interpolation=cv2.INTER_CUBIC)
    # image=cv2.imread('C:\\Users\\HP\\Desktop\\NTU\\FYP\\FYP Code\\MAAC\\utils\\bomb.jpg')
    # gl=cv2.resize(image, dsize=(40, 40), interpolation=cv2.INTER_CUBIC)

    # create bg
    img = np.ones((400, 400, 3), dtype=np.uint8)*0
    try:
        # create vertical and horizontal boundary
        img[199:201, :, :] = 255
        img[:, 199:201, :] = 255

        # indicate agent
        a_x = agent_loc[0]
        a_y = agent_loc[1]
        #img[a_x-3:a_x+3, a_y-3:a_y+3] = [255, 0, 0]
        img[a_x-3:a_x+3, a_y-3:a_y+3] = [0,0,255]

        # indicate goal
        g_x = goal_loc[0]
        g_y = goal_loc[1]
        img[g_x-3:g_x+3, g_y-3:g_y+3] = [0,255,0]


        for ghost_loc in ghosts_locs:
            
            gh_x = ghost_loc[0]
            gh_y = ghost_loc[1]

            r = 10
            r2 = 8
            x = np.arange(0, 400)
            y = np.arange(0, 400)
            mask = (x[np.newaxis,:]-gh_y)**2 + (y[:,np.newaxis]-gh_x)**2 < r**2
            img[mask] = [220,220,220]

            img[gh_x-3:gh_x+3, gh_y-3:gh_y+3] = [255,0,0]

        # circle = (g_x - 100) ** 2 + (g_y - 100) ** 2
        # donut = np.logical_and(circle < (6400 + 60), circle > (6400 - 60))
        # m1 = ((g_y-200)**2 + (g_x-100)**2 < 30**2)
        # m2 = ((g_y-350)**2 + (g_x-400)**2 < 20**2)
        # m3 = ((g_y-260)**2 + (g_x-200)**2 < 20**2)
        # img[m1+m2+m3]=1

        # radius = 100
        # cx, cy = 1, 1 # The center of circle
        # y, x = np.ogrid[-radius: radius, -radius: radius]
        # index = x**2 + y**2 <= radius**2
        # img[cy-radius:cy+radius, cx-radius:cx+radius][index] = [0,255,0]
        
        # x = np.linspace(-2, 2, 800)
        # y = np.linspace(-2, 2, 800)
        # mask = np.sqrt((x-g_x)**2+(y-g_y)**2)

    except:
        pass
    #return (img)
    return np.array([img])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.imshow(get_image((1, 1), (-0.5, 0.5),[[0.1,-0.9],[-0.2,-0.8]])[0])
    plt.show()

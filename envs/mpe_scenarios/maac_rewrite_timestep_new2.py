# Trying to visualise the predicted rewards for the adversarial agent

# Current attempt --> Add in another learner agent to simply stay put 
# but receive the rewards of the adversarial lad

import numpy as np
import seaborn as sns
import random

#Importing from multiagent package
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


# AGENT TYPES

AGENT_LISTENER = 0
AGENT_SPEAKER = 1
AGENT_GHOST = 2
random_choices = [[0.03, 0],[-0.03, 0],[0, -0.03],[0, 0.03]]
def world_step(world):
    for ghost in world.ghosts:
        dir = random.choice(random_choices)
        ghost.state.p_pos[0] += dir[0]
        ghost.state.p_pos[1] += dir[1]
    World.step(world)

class Scenario(BaseScenario):

    #Creates all of the entities that inhabit the world (landmarks, agents, etc.)
    #Assigns capabilities of entities (whether they can communicate, or move, or both)
    #Called once at the beginning of each training session

    def make_world(self):
        world = World()
        world.step = lambda :world_step(world)

        #Setting World Properties
        world.dim_c = 5
        self.num_listeners = 2 # Number of listeners= Number of agents
        self.num_speakers = 4  # Number of speakers= Number of advisors
        self.num_landmarks = 1 # Number of landmarks= Number of goals
        self.num_ghosts = 4

        world.landmark_colors = np.array(sns.color_palette(n_colors=self.num_landmarks))

        #Creation of listeners
        world.listeners = []
        for listener_ind in range(self.num_listeners):
            agent = self._create_listener(listener_ind)
            world.listeners.append(agent)
        world.listeners[-1].predicted = True

        #Creation of speakers
        world.speakers = []
        for speaker_ind in range(self.num_speakers):
            agent = self._create_speaker(speaker_ind)
            world.speakers.append(agent)
        world.speakers[-1].predicted = True

        world.landmarks = []
        for landmark_ind in range(self.num_landmarks):
            landmark = self._create_landmark(landmark_ind)
            landmark.color = world.landmark_colors[landmark_ind]
            world.landmarks.append(landmark)

        world.ghosts = []
        for ghost_ind in range(self.num_ghosts):
            agent = self._create_ghost(ghost_ind)
            world.ghosts.append(agent)

        #The World is collectively made up of listeners and speakers
        world.agents = world.listeners + world.speakers

        #Set initial conditions
        self.reset_world(world)
        self.reset_cached_rewards()
        return world

    def _create_landmark(self, ind):
        landmark = Landmark()
        ind = ind + self.num_speakers + self.num_listeners
        landmark.i = ind
        landmark.name = 'landmark %d' % ind
        landmark.collide = False
        landmark.movable = False
        landmark.size = 0.04
        return landmark

    def _create_agent_base(self, ind):
        agent = Agent()
        agent.i = ind
        agent.name = 'agent %i' % agent.i
        agent.collide = False
        agent.size = 0.075
        agent.accel = 1.5
        agent.initial_mass = 1.0
        agent.max_speed = 1.0
        agent.predicted=False
        return agent

    def _create_speaker(self, ind):
        ind = ind + self.num_listeners
        agent = self._create_agent_base(ind)
        agent.listener = False
        agent.agent_type = AGENT_SPEAKER
        agent.movable = False
        agent.predicted=False
        return agent

    def _create_listener(self, ind):
        agent = self._create_agent_base(ind)
        agent.agent_type = AGENT_LISTENER
        agent.listener = True
        agent.silent = True
        agent.movable = True
        return agent

    def _create_ghost(self, ind):
        ind = ind + self.num_speakers + self. num_landmarks + self.num_listeners
        agent = self._create_agent_base(ind)
        agent.listener = False
        agent.agent_type = AGENT_GHOST
        agent.movable = True
        return agent

    def reset_cached_rewards(self):
        self.pair_rewards = None

    def post_step(self, world):
        self.reset_cached_rewards()

    #method to signal end of game
    def game_done(self, agent, world):
        return world.done
    
    @staticmethod
    def get_quadrant(postion):
        if postion[0]>0:
            if postion[1]>0:
                return 0
            else:
                return 3
        else:
            if postion[1]>0:
                return 1
            else:
                return 2

    def _reset_movable(self, world):
        for agent in world.listeners + world.speakers + world.ghosts:
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

    def _reset_landmarks(self, world):
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def _reset_speakers(self, world, shuff=True):
        landmark = np.random.choice(world.landmarks)
        quads = np.arange(4)
        if shuff:
            np.random.shuffle(quads)
        for i, speaker in enumerate(world.speakers):
            li = 0
            speaker.listen_ind = li
            speaker.goal_a = world.listeners[li]
            speaker.goal_b = landmark
            speaker.quad = quads[i]
            speaker.color = np.array([0.25, 0.25, 0.25])
            world.listeners[li].color = speaker.goal_b.color + np.array([0.25, 0.25, 0.25])
            world.listeners[li].speak_ind = i
            # speaker.state.c = np.zeros(world.dim_c)

    def reset_world(self, world):
        # listen_inds = list(range(len(world.listeners)))
        # np.random.shuffle(listen_inds)  #Randomize which listener is used every episode
        self._reset_speakers(world)
        self._reset_movable(world)
        #   #Initial states are set at random
        #   for agent in world.agents:
        #       agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
        #       agent.state.p_vel = np.zeros(world.dim_p)
        #       agent.state.c = np.zeros(world.dim_c)

        self._reset_landmarks(world)
        self.reset_cached_rewards()
        
        world.done= False

    def benchmark_data(self, agent, world):
        #Data returned for benchmarking purposes
        return reward(agent, world)

    def calc_rewards(self, world):
        rews = []
        for speaker in world.speakers:
            dist = np.sum(np.square(speaker.goal_a.state.p_pos -
                                    speaker.goal_b.state.p_pos))
            rew = -dist
            if dist < (speaker.goal_a.size + speaker.goal_b.size) * 1.5:
                world.done = True
            
            for ghost in world.ghosts:
                dist = np.sum(np.square(speaker.goal_a.state.p_pos -
                        ghost.state.p_pos))
                if dist< 0.225:
                    if dist>0.001:
                        rew -= (0.01)* (1/dist)
                    else:
                        rew -= 10
            if speaker.predicted:
                rew += 1
            else:
                rew -= 1
                    
            rews.append(rew)
        # print (rews)
        return rews

    def reward(self, agent, world):
        if self.pair_rewards is None:
            self.pair_rewards = self.calc_rewards(world)
        share_rews = False
        if share_rews:
            return sum(self.pair_rewards)
        if agent.listener:
            if agent.predicted:
                return self.pair_rewards[-1]
            else:
                return self.pair_rewards[-2]
        else:
            return self.pair_rewards[-2]

    def observation(self, agent, world):
        if agent.listener:
            obs = []
            #Listener gets index of speaker
            # obs += [agent.speak_ind == np.arange(len(world.speakers))]
            #Listener gets communication from speaker
            #obs += [world.speakers[agent.speak_ind].state.c]
            #Listener gets own position or velocity
            obs += [agent.state.p_pos, agent.state.p_vel]

            # obs += [world.speakers[agent.speak_ind].state.c]
            # # # give listener index of their speaker
            # # obs += [agent.speak_ind == np.arange(len(world.speakers))]
            # give listener all communications
            obs += [speaker.state.c for speaker in world.speakers]
            # # give listener its own velocity
            # obs += [agent.state.p_vel]
            # # give listener locations of all agents
            # # obs += [a.state.p_pos for a in world.agents]
            # # give listener locations of all landmarks
            # obs += [l.state.p_pos for l in world.landmarks]
            return np.concatenate(obs)

        else: #If agent is a speaker
            obs = []
            #Speaker gets index of their listener
            # obs += [agent.listen_ind == np.arange(len(world.listeners))]
            # Speaker gets position and goal of listener
            if agent.quad == self.get_quadrant(agent.goal_a.state.p_pos):
                obs += [np.array([1]), agent.goal_a.state.p_pos, agent.goal_b.state.p_pos]
            else:
                obs += [np.array([0,0,0]), agent.goal_b.state.p_pos]

            for ghost in world.ghosts:
                if agent.quad == self.get_quadrant(ghost.state.p_pos):
                    obs += [np.array([1]), ghost.state.p_pos]
                else:
                    obs += [np.array([0,0,0])]

            # # give speaker index of their listener
            # # obs += [agent.listen_ind == np.arange(len(world.listeners))]
            # # # give speaker all communications
            # # obs += [speaker.state.c for speaker in world.speakers]
            # # give speaker their goal color
            # obs += [agent.goal_b.color]
            # # give speaker their listener's position
            # obs += [agent.goal_a.state.p_pos]
            #
            # obs += [speaker.state.c for speaker in world.speakers]
            return np.concatenate(obs)

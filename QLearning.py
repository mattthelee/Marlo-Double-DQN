import marlo
import numpy as np
import random
import json

#import MalmoPython
#TODO - Is this needed??

class QLearningAgent(object):

    def __init__(self, actions, epsilon=0.9, alpha=0.1, gamma=1.0 ):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.training = True

        self.actions = [i for i in range(actions)]
        # How to find what actions are available in the particular mission?
        self.q_table = {}

    #TODO - Load the model from a JSON
    def loadModel(self, model_file):
        """load q table from model_file"""
        with open(model_file) as f:
            self.q_table = json.load(f)


    def runAgent(self, env):

        for i in range(1000):

            obs = env.reset()

            action = env.action_space.sample()
            obs, currentReward, done, info = env.step(action)

            print(info)
            # TODO - Is this the correct way to get the initial info?

            #TODO - Function to discretise the info and just return a Q-Table Key
            currentState = "%d:%d:%d" % (int(info['observation']["XPos"]), int(info['observation']["YPos"]), int(info['observation']["Yaw"]))

            print("currentState: " + currentState)

            while not done:
                done, newState = self.act(env, currentState)

                currentState = newState



    def act(self, env, currentState):

        # If no Q Value for this state, Initialise
        if currentState not in self.q_table:
            self.q_table[currentState] = ([0] * len(self.actions))

        # select the next action
        if random.random() < self.epsilon:
            # Choose a random action
            action = random.randint(0, len(self.actions) - 1)
            print("From State %s, taking random action: %s" % (currentState, self.actions[action]))
        else:
            # Pick the highest Q-Value action for the current state
            currentStateActions = self.q_table[currentState]

            print('currentStateActions: ' + str(currentStateActions))

            action = random.choice(np.nonzero(currentStateActions == np.amax(currentStateActions))[0])
            # Pick highest action Q-value - randomly cut ties

            print("From State %s, taking q action: %s" % (currentState,  self.actions[action]))

        obs, currentReward, done, info = env.step(action)
        #TODO - Check this reward - what is it??

        #TODO - Can we get this action to move a whole block

        #TODO - Function to discretise the info and just return a Q-Table Key


        if done:
            return done, currentState


        newState = "%d:%d:%d" % (int(info['observation']["XPos"]), int(info['observation']["YPos"]),int(info['observation']["Yaw"]))
        # If no Q Value for this state, Initialise
        if newState not in self.q_table:
            self.q_table[newState] = ([0] * len(self.actions))

        print('Q-Value for Current State: ')
        print(self.q_table[currentState])

        # update Q-values for this action
        if self.training:
            oldQValueAction = self.q_table[currentState][action]

            self.q_table[currentState][action] = oldQValueAction + self.alpha * (currentReward + self.gamma * max(
                self.q_table[newState]) - oldQValueAction)

        return done, newState



def loadMissionFile(filename):
    with open(filename, 'r') as file:
        missionXML = file.read()
    return missionXML

def main():
    client_pool = [('127.0.0.1', 10000)]
    # Suppress info set to false to allow the agent to train using additional features, this will be off for testing!
    join_tokens = marlo.make('MarLo-FindTheGoal-v0', params={"client_pool": client_pool, 'suppress_info': False})
    assert len(join_tokens) == 1
    join_token = join_tokens[0]

    env = marlo.init(join_token)
    # Change the spec of the mission by loading xml from file
    missionXML= loadMissionFile('find_the_goal_mission2.xml')
    #missionXML= loadMissionFile('mission_spec')

    # Get the number of available actions
    actionSize = env.action_space.n
    #TODO - Is this the best way to encode the actions??

    myAgent = QLearningAgent(actionSize)

    myAgent.runAgent(env)

    #env.mission_spec = MalmoPython.MissionSpec(missionXML, True)
    #TODO - See how to do this?



    return


if __name__ == "__main__":
    main()
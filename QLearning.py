import marlo
import numpy as np
import random
import json

import os
os.chdir("C:/Users/benja/OneDrive/Documents/Queen Mary/Artificial Intelligence in Games/Assignments/MARLO/marlo")
import utils


#TODO - How to Save and Load the models
#TODO - Keeps getting Mission ended: command_quota_reached;command_quota_reached - Is this ok?

class QLearningAgent(object):

    def __init__(self, actions, epsilon=0.1, alpha=0.1, gamma=1.0 ):
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

            # Initialise the MineCraft environment
            obs = env.reset()
            # Do an initial 'stop' step in order to get info from env
            obs, currentReward, done, info = env.step(0)

            print("Info: ")
            print(info)

            # Use utils module to discrete the info from the game
            [xdisc, ydisc, zdisc, yawdisc, pitchdisc] = utils.discretiseState(info)
            currentState = "%d:%d:%d" % (int(xdisc), int(zdisc), int(yawdisc))
            print("currentState: " + currentState)

            while not done:
                done, newState, newInfo = self.act(env, currentState, info)
                currentState = newState
                info = newInfo



    def act(self, env, currentState, info):

        # If no Q Value for this state, Initialise
        if currentState not in self.q_table:
            self.q_table[currentState] = ([0] * len(self.actions))

        # Select the next action
        if random.random() < self.epsilon:
            # Choose a random action
            action = random.randint(0, len(self.actions) - 1)
            print("From State %s, taking random action: %s" % (currentState, self.actions[action]))
        else:
            # Pick the highest Q-Value action for the current state
            currentStateActions = self.q_table[currentState]

            print('currentStateActions: ' + str(currentStateActions))

            # Pick highest action Q-value - randomly cut ties
            action = random.choice(np.nonzero(currentStateActions == np.amax(currentStateActions))[0])

            print("From State %s, taking q action: %s" % (currentState,  self.actions[action]))


        # Plays an action to move a full block or turn a full 45 degrees
        new_state, currentReward, done, info = utils.discreteMove(env, action, info)

        #TODO - Check this reward - what is it??


        # Check if the game is finished
        if done:
            print('------- Game Finished ----------')
            return done, currentState, info

        # Use utils module to discrete the info from the game
        [xdisc, ydisc, zdisc, yawdisc, pitchdisc] = utils.discretiseState(info)
        newState = "%d:%d:%d" % (int(xdisc), int(zdisc), int(yawdisc))


        # If no Q Value for this state, initialise as all 0's
        if newState not in self.q_table:
            self.q_table[newState] = ([0] * len(self.actions))

        print('Q-Value for Current State: ')
        print(self.q_table[currentState])

        # update Q-values for this action
        if self.training:
            oldQValueAction = self.q_table[currentState][action]

            self.q_table[currentState][action] = oldQValueAction + self.alpha * (currentReward + self.gamma * max(
                self.q_table[newState]) - oldQValueAction)

        return done, newState, info



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
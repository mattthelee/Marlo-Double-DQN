import marlo
import numpy as np
import random
import json
import utils


#TODO - Keeps getting Mission ended: command_quota_reached;command_quota_reached - Is this ok?

#TODO - Check the choosing of the actions and what is given by the map

#TODO - How to load different missions?


class QLearningAgent(object):

    def __init__(self, actions, loadQTable = False, epsilon=0.1, alpha=0.1, gamma=1.0 ):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.training = True

        self.actions = [i for i in range(actions)]
        # How to find what actions are available in the particular mission?


        if loadQTable:
            # Load the Q-Table from a JSON
            QTableFile = 'QTable.json'
            with open(QTableFile) as f:
                self.qTable = json.load(f)
        else:
            # Initialise the Q-Table from blank
            self.qTable = {}

    #TODO - Load the model from a JSON

    def runAgent(self, env):

        for i in range(1000):

            print(" ------- New Game ----------  \n")

            #Store the Q-Table as a JSON
            print("Saving QTable as JSON")
            with open('QTable.json', 'w') as fp:
                json.dump(self.qTable, fp)

            if (i+1) % 10 == 0:
                print("Saving QTable BackUp as JSON")
                # Store a QTable BackUp too every 10 games
                with open('QTableBackUp.json', 'w') as fp:
                    json.dump(self.qTable, fp)

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
        if currentState not in self.qTable:
            self.qTable[currentState] = ([0] * len(self.actions))

        # Select the next action
        if random.random() < self.epsilon:
            # Choose a random action
            action = random.randint(0, len(self.actions) - 1)
            print("From State %s (X,Z,Yaw), taking random action: %s" % (currentState, self.actions[action]))
        else:
            # Pick the highest Q-Value action for the current state
            currentStateActions = self.qTable[currentState]

            print('currentStateActions: ' + str(currentStateActions))

            # Pick highest action Q-value - randomly cut ties
            action = random.choice(np.nonzero(currentStateActions == np.amax(currentStateActions))[0])

            print("From State %s (X,Z,Yaw), taking q action: %s" % (currentState,  self.actions[action]))


        # Plays an action to move a full block or turn a full 45 degrees
        new_state, currentReward, done, info = utils.discreteMove(env, action, info)

        #TODO - Check this reward - what is it??


        # Check if the game is finished
        if done:
            print(' ------- Game Finished ----------  \n')
            return done, currentState, info

        # Use utils module to discrete the info from the game
        [xdisc, ydisc, zdisc, yawdisc, pitchdisc] = utils.discretiseState(info)
        newState = "%d:%d:%d" % (int(xdisc), int(zdisc), int(yawdisc))


        # If no Q Value for this state, initialise as all 0's
        if newState not in self.qTable:
            self.qTable[newState] = ([0] * len(self.actions))

        print('Q-Value for Current State: ')
        print(self.qTable[currentState])

        # update Q-values for this action
        if self.training:
            oldQValueAction = self.qTable[currentState][action]

            self.qTable[currentState][action] = oldQValueAction + self.alpha * (currentReward + self.gamma * max(
                self.qTable[newState]) - oldQValueAction)

        return done, newState, info



def main():
    env = utils.setupEnv('find_the_goal_mission2.xml')
    # Get the number of available actions
    actionSize = env.action_space.n

    # Give user decision on loadind model or not
    load = input("Load Q Table? y/n - Default as y:________")

    # Set the Agent to Load Q-Table if user chooses
    if load.lower() == 'n':
        myAgent = QLearningAgent(actionSize)
    else:
        myAgent = QLearningAgent(actionSize, True)

    # Start the running of the Agent
    myAgent.runAgent(env)
    return


if __name__ == "__main__":
    main()

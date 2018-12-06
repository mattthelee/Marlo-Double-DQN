import marlo
import numpy as np
import random
import json
import utils
import  csv
import sys


#TODO - How to load different missions? Answer: see the utils.setupEnv function

class QLearningAgent(object):

    def __init__(self, actions, QTableName = 'QTable.json', CSVName = 'qlearningResults.csv', loadQTable = False, epsilon_decay=0.99, alpha=0.1, gamma=0.9 ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_min = 0.01
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.training = True if self.epsilon > 0 else False
        # Don't consider waiting action
        self.actions = [i for i in range(1,actions)]
        self.QTableName = QTableName
        self.CSVName = CSVName

        if loadQTable:
            # Load the Q-Table from a JSON
            QTableFile = self.QTableName
            with open(QTableFile) as f:
                self.qTable = json.load(f)
        else:
            # Initialise the Q-Table from blank
            self.qTable = {}
        return

    def runAgent(self,env):
        results = []
        for i in range(200):
            print("Game " + str(i))
            currentState, info = self.startGame(env,i)
            actionCount = 0
            score = 0
            done  = False
            while not done:
                # Chose the action then run it
                action = self.act(env, currentState, info)
                obs, reward, done, info = env.step(action)
                # Continue counts of actions and scores
                actionCount += 1
                score += reward
                # Check if game is done
                if done:
                    break
                # have to use this to keep last info for results
                oldInfo = info
                # Use utils module to discrete the info from the game
                [xdisc, ydisc, zdisc, yawdisc, pitchdisc] = utils.discretiseState(info['observation'])
                newState = "%d:%d:%d:%d:%d" % (xdisc, zdisc, yawdisc, ydisc, pitchdisc)

                print('Q-Value for Current State: ')
                print(self.qTable[currentState])

                # If no Q Value for this state, Initialise
                if newState not in self.qTable:
                    self.qTable[newState] = ([0] * len(self.actions))

                # update Q-values for this action
                if self.training:
                    oldQValueAction = self.qTable[currentState][self.actions.index(action)]
                    self.qTable[currentState][self.actions.index(action)] = oldQValueAction + self.alpha * (reward + self.gamma * max(
                        self.qTable[newState]) - oldQValueAction)
                currentState = newState

            print('\n ------- Game Finished ----------  \n')
            results.append([score,actionCount,oldInfo['observation']['TotalTime'], self.epsilon])
            # Decay the epsilon until the minimum
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            else:
                # Will take 458 rounds
                self.epsilon = 0
            with open(self.CSVName,"w") as f:
                wr = csv.writer(f)
                wr.writerows(results)

        return results


    def startGame(self,env, i):
        print(" ------- New Game ----------  \n")
        #Store the Q-Table as a JSON
        print("Saving QTable as JSON")
        with open(self.QTableName, 'w') as fp:
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

        # Use utils module to discretise the info from the game
        [xdisc, ydisc, zdisc, yawdisc, pitchdisc] = utils.discretiseState(info['observation'])
        currentState = "%d:%d:%d:%d:%d" % (xdisc, zdisc, yawdisc, ydisc, pitchdisc)
        print("initialState: " + currentState)
        return currentState, info


    def act(self, env, currentState, info):

        # If no Q Value for this state, Initialise
        if currentState not in self.qTable:
            self.qTable[currentState] = ([0] * len(self.actions))

        # Select the next action
        if random.random() < self.epsilon:
            # Choose a random action
            action = random.choice(self.actions)
            print("From State %s (X,Z,Yaw,Y,Pitch), taking random action: %s" % (currentState, action))
        else:
            # Pick the highest Q-Value action for the current state
            currentStateActions = self.qTable[currentState]

            print('currentStateActionsQValues: ' + str(currentStateActions))

            # Pick highest action Q-value - In case of tie (very unlikely) chooses first in list
            action = self.actions[np.argmax(currentStateActions)]

            print("From State %s (X,Z,Yaw,Y,Pitch), taking q action: %s" % (currentState,  action))
        return action


def main():
    if len(sys.argv) > 1:
        env = utils.setupEnv(sys.argv[1])
    else:
        env = utils.setupEnv()
    # Get the number of available actions
    actionSize = env.action_space.n

    # Give user decision on loadind model or not
    load = input("Load Q Table? y/n - Default as y:________")

    # Set the Agent to Load Q-Table if user chooses to load
    if load.lower() == 'n':
        myAgent = QLearningAgent(actionSize,'QTable.json', 'qlearningResults.csv' )
    else:
        myAgent = QLearningAgent(actionSize, 'QTable.json', 'qlearningResults.csv' , True)

    # Start the running of the Agent
    myAgent.runAgent(env)

    return

if __name__ == "__main__":
    main()

import marlo
import numpy as np
import random
import json
import utils
import csv
import sys

#TODO chnage this into mc

class MC_agent(object):

    def __init__(self, actions, QTableName = 'mc_QTable.json', CSVName = 'mc_qlearningResults.csv',loadQTable = False, epsilon_decay=0.99, epsilon=1.0, alpha=0.1, gamma = 0.9, training = True):
        self.alpha = alpha
        self.epsilon_min = 0.01
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.training = training
        self.QTableName = QTableName
        self.CSVName = CSVName

        # Don't consider waiting action
        self.actions = [i for i in range(1,actions)]

        if loadQTable:
            # Load the Q-Table from a JSON
            mc_QTableFile = 'mc_QTable.json'
            with open(mc_QTableFile) as f:
                self.qTable = json.load(f)
        else:
            # Initialise the Q-Table from blank
            self.qTable = {}
        return



    def startGame(self,env, i):
        print(" ------- New Game ----------  \n")
        #Store the Q-Table as a JSON
        print("Saving mc_QTable as JSON")
        with open(self.QTableName, 'w') as fp:
            json.dump(self.qTable, fp)
        if (i+1) % 10 == 0:
            print("Saving mc_QTable BackUp as JSON")
            # Store a QTable BackUp too every 10 games
            with open('mc_QTableBackUp.json', 'w') as fp:
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


    def runAgent(self,env):
        results = []
        states_count = {}

        for i in range(200):
            print("Game " + str(i))
            currentState, info = self.startGame(env,i)
            actionCount = 0
            score = 0
            done  = False
            history = []

            while not done:
                # Chose the action then run it
                action = self.act(env, currentState)
                image, reward, done, obs = utils.completeAction(env,action)
                print(f"Reward of {reward}")
                # Continue counts of actions and scores
                actionCount += 1
                score += reward

                if done:
                    if self.training:
                        oldQValueAction = self.qTable[currentState][self.actions.index(action)]
                        self.qTable[currentState][self.actions.index(action)] = oldQValueAction + self.alpha * (
                                    reward - oldQValueAction)
                    break
                # have to use this to keep last info for results
                oldObs = obs
                # Use utils module to discrete the info from the game
                [xdisc, ydisc, zdisc, yawdisc, pitchdisc] = utils.discretiseState(obs)
                newState = "%d:%d:%d:%d:%d" % (xdisc, zdisc, yawdisc, ydisc, pitchdisc)

                if newState not in states_count:
                    states_count[newState] = ([0] * len(self.actions))

                history.append([newState,action,reward])
                states_count[newState][self.actions.index(action)] += 1.0



                # Check if game is done

                print('Q-Value for Current State: ')
                print(self.qTable[currentState])

                # If no Q Value for this state, Initialise
                if newState not in self.qTable:
                    self.qTable[newState] = ([0] * len(self.actions))

            for t,[ep_state,ep_action,reward] in enumerate(history):
            # update Q-values for this action
                return_val = reward + sum([ x[2] * self.gamma ** i for i , x in enumerate(history[t:])])
                if self.training:
                    oldQValueAction = self.qTable[ep_state][self.actions.index(ep_action)]
                    self.qTable[ep_state][self.actions.index(ep_action)] = oldQValueAction + 1/states_count[ep_state][self.actions.index(ep_action)] * \
                                                                              (return_val - oldQValueAction)


            print(' ------- Game Finished ----------  \n')
            results.append([score,actionCount,oldObs['TotalTime'], self.epsilon])
            # Decay the epsilon until the minimum
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            else:
                self.epsilon = 0
            with open(self.CSVName,"w") as f:
                wr = csv.writer(f)
                wr.writerows(results)
        return results


    def act(self, env, currentState):


        # If no Q Value for this state, Initialise
        if currentState not in self.qTable:
            self.qTable[currentState] = ([0] * len(self.actions))

        # Select the next action
        if random.random() < self.epsilon:
            # Choose a random action
            action = random.choice(self.actions)
            print("From State %s (X,Z,Yaw), taking random action: %s" % (currentState, action))
        else:
            # Pick the highest Q-Value action for the current state
            currentStateActions = self.qTable[currentState]

            print('currentStateActionsQValues: ' + str(currentStateActions))

            # Pick highest action Q-value - In case of tie (very unlikely) chooses first in list
            action = self.actions[np.argmax(currentStateActions)]

            print("From State %s (X,Z,Yaw), taking q action: %s" % (currentState,  action))
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

    # Set the Agent to Load Q-Table if user chooses
    if load.lower() == 'n':
        myAgent = MC_agent(actionSize)
    else:
        myAgent = MC_agent(actionSize, True)

    # Start the running of the Agent
    myAgent.runAgent(env)

    return

if __name__ == "__main__":
    main()

import marlo
import numpy as np
import random
import json
import utils
import csv
import sys
import time
# Initialise class
class MC_agent(object):

    def __init__(self, actions, mc_QTableName = 'mc_QTable.json', mc_CSVName = 'mc_qlearningResults.csv',loadQTable = False, epsilon_decay=0.99, epsilon=1.0, alpha=0.1, gamma = 0.9, training = True):
        # Initialise parameters
        self.alpha = alpha
        self.epsilon_min = 0.01
        self.gamma = gamma
        self.epsilon = 1
        self.epsilon_decay = epsilon_decay
        self.training = training
        self.mc_QTableName = mc_QTableName
        self.mc_CSVName = mc_CSVName


        # Don't consider waiting action
        self.actions = [i for i in range(1,actions)]

        if loadQTable:
            # Load the Q-Table from a JSON
            mc_QTableFile = 'mc_QTable.json'
            with open(mc_QTableFile) as f:
                self.mc_qTable = json.load(f)
        else:
            # Initialise the Q-Table from blank
            self.mc_qTable = {}
        return

    def startGame(self,env, i):
        # Perform startup process for game
        print(" ------- New Game ----------  \n")
        #Store the Q-Table as a JSON
        print("Saving mc_QTable as JSON")
        with open(self.mc_QTableName, 'w') as fp:
            json.dump(self.mc_qTable, fp)
        if (i+1) % 10 == 0:
            print("Saving mc_QTable BackUp as JSON")
            # Store a QTable BackUp  every 10 games
            with open('mc_QTableBackUp.json', 'w') as fp:
                json.dump(self.mc_qTable, fp)
        # Perform sleep as this has reduce Marlo disconnects
        time.sleep(5)
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
        # Run GLIE-MC algorithm
        results = []
        states_count = {}
        # Train for 200 games
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
                image,reward,done, info = env.step(action)
                obs = info['observation']

                # Continue counts of actions and scores
                actionCount += 1
                score += reward
                # Store history of this episode for later q-value calcs
                history.append([currentState, action, reward])
                # If we've not visited this state before, initialise it's dictionary
                if currentState not in states_count:
                    states_count[currentState] = ([0] * len(self.actions))
                # Increment state-action count
                states_count[currentState][self.actions.index(action)] += 1.0

                if done:
                    break

                # have to use this to keep last info for results
                oldObs = obs
                # Use utils module to discrete the info from the game
                [xdisc, ydisc, zdisc, yawdisc, pitchdisc] = utils.discretiseState(obs)
                newState = "%d:%d:%d:%d:%d" % (xdisc, zdisc, yawdisc, ydisc, pitchdisc)

                # If no Q Value for this state, Initialise
                if newState not in self.mc_qTable:
                    self.mc_qTable[newState] = ([0] * len(self.actions))
                currentState = newState

            # For each step of the episode memory
            for t,[ep_state,ep_action,reward] in enumerate(history):
            # update Q-values for this action
                # Calculate the return
                return_val = reward + sum([ x[2] * self.gamma ** i for i , x in enumerate(history[t:])])
                if self.training:
                    # Perform GLIE-MC update
                    oldQValueAction = self.mc_qTable[ep_state][self.actions.index(ep_action)]
                    self.mc_qTable[ep_state][self.actions.index(ep_action)] = oldQValueAction + 1/states_count[ep_state][self.actions.index(ep_action)] * \
                                                                              (return_val - oldQValueAction)


            print(' ------- Game Finished ----------  \n')
            # write results to file for later analysis
            results.append([score,actionCount,oldObs['TotalTime'], self.epsilon])
            with open(self.mc_CSVName,"w") as f:
                wr = csv.writer(f)
                wr.writerows(results)
            # Decay the epsilon until the minimum
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            else:
                self.epsilon = 0
        return results


    def act(self, env, currentState):
        # Get the epsilon-greedy action for this state
        # If no Q Value for this state, Initialise
        if currentState not in self.mc_qTable:
            self.mc_qTable[currentState] = ([0] * len(self.actions))

        # Select the next action
        if random.random() < self.epsilon:
            # Choose a random action
            action = random.choice(self.actions)
            print("From State %s (X,Z,Yaw), taking random action: %s" % (currentState, action))
        else:
            # Pick the highest Q-Value action for the current state
            currentStateActions = self.mc_qTable[currentState]
            # Pick highest action Q-value - In case of tie (very unlikely) chooses first in list
            action = self.actions[np.argmax(currentStateActions)]
            print("From State %s (X,Z,Yaw), taking q action: %s" % (currentState,  action))
        return action

def main():
    # Takes in command line arguments for setup of environment
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

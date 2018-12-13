import marlo
import numpy as np
import random
import json
import utils
import csv
from time import sleep
import sys


# Main Q-Learning Agent Class
class QLearningAgent(object):

    # Initialisation of the Class
    def __init__(self, actions, episodes = 200, QTableName = 'QTable.json', CSVName = 'qlearningResults.csv', loadQTable = False, epsilon_decay=0.99, alpha=0.5, gamma=1,epsilon = 1.0, training = True ):
        self.alpha = alpha # Given Alpha
        self.gamma = gamma # Given Gamma
        self.epsilon_min = 0.1 # Set Epsilon-Min,
        self.epsilon = epsilon # Given Epsilon
        self.epsilon_decay = epsilon_decay # Given Epsilon Decay
        self.training = training # Given training flag
        self.actions = [i for i in range(1,actions)] # Given action space - Don't consider waiting action
        self.QTableName = QTableName # Given QTable Name - To save QTable as JSON
        self.CSVName = CSVName # Given CSV Name = To save results
        self.episodes = episodes # Given episodes

        if loadQTable:
            # Load the Q-Table from a JSON with the provided file name
            QTableFile = self.QTableName
            with open(QTableFile) as f:
                self.qTable = json.load(f)
        else:
            # Initialise the Q-Table from blank
            self.qTable = {}
        return

    # Function to run the Q-Learning Agent, for the given amount of episodes
    # Given a environment to run on
    def runAgent(self,env):
        results = []
        # Run for the amount of episodes provided
        for i in range(self.episodes):
            print("Game: " + str(i + 1))
            print("Epsilon: " + str(self.epsilon))
            print("Training: " + str(self.training))
            # Start the game using the 'startGame' function
            currentState, obs = self.startGame(env,i)
            # Initialise actions, score and done boolean
            actionCount = 0
            score = 0
            done  = False

            # Loop through taking actions until the game is done
            while not done:
                # Chose an action using the 'act' function then run it
                action = self.act(env, currentState)

                # Play the given action in the environment
                image, reward, done, info = env.step(action)

                # Get the observations from the info provided
                obs = info['observation']
                # Continue counts of actions and scores
                actionCount += 1
                score += reward

                # Check if game is done, if so, update the Q-Table and stop the game
                if done:
                    # update Q-values for this action
                    if self.training:
                        # Find the Q-Functions of the current state and print to screen
                        currentStateActions = self.qTable[currentState]
                        print('\nCurrentStateActionsQValues: ' + str(currentStateActions))

                        # Find the Q-Function of this current state-action pair
                        oldQValueAction = self.qTable[currentState][self.actions.index(action)]
                        # Update this Q-Function following the Bellman Equation
                        self.qTable[currentState][self.actions.index(action)] = oldQValueAction + self.alpha * (reward - oldQValueAction)
                        print("Reward of %s added to the Q-Table at %s with action %s" % (str(reward), currentState, action))

                        # Find the new Q-Functions for this state and print to screen
                        currentStateActions = self.qTable[currentState]
                        print('Updated CurrentStateActionsQValues: ' + str(currentStateActions))
                        newQValueAction = self.qTable[currentState][self.actions.index(action)]
                        print("Q-Value difference for action %s of %s" % (action, abs(oldQValueAction - newQValueAction)))
                        print("\n -------- Final Score: -------- %s" % (score))

                    break

                # Use this to keep last info for results
                oldObs = obs
                # Use utils module to discrete the info from the game
                [xdisc, ydisc, zdisc, yawdisc, pitchdisc] = utils.discretiseState(obs)
                # Find the position of this new state
                newState = "%d:%d:%d:%d:%d" % (xdisc, zdisc, yawdisc, ydisc, pitchdisc)


                # If no Q-Function for this state in the Q-Table, initialise it
                if newState not in self.qTable:
                    self.qTable[newState] = ([0] * len(self.actions))


                # Update Q-values for this action, if training is set to True
                if self.training:
                    # Find the Q-Functions of the current state and print to screen
                    currentStateActions = self.qTable[currentState]
                    print('\nCurrentStateActionsQValues: ' + str(currentStateActions))

                    # Find the Q-Function of this current state-action pair
                    oldQValueAction = self.qTable[currentState][self.actions.index(action)]
                    # Update this Q-Function following the Bellman Equation
                    self.qTable[currentState][self.actions.index(action)] = oldQValueAction + self.alpha * (reward + self.gamma * max(self.qTable[newState]) - oldQValueAction)
                    print("Reward of %s added to the Q-Table at %s with action %s" % (str(reward), currentState, action))


                    # Find the new Q-Functions for this state and print to screen
                    currentStateActions = self.qTable[currentState]
                    print('Updated CurrentStateActionsQValues: ' + str(currentStateActions))
                    newQValueAction = self.qTable[currentState][self.actions.index(action)]
                    print("Q-Value difference for action %s of %s"%(action,abs(oldQValueAction-newQValueAction)))

                # Move to the new current state, ready to take the next action
                currentState = newState

            print('\n ------- Game Finished ----------  \n')
            # Store the results of this run - If the 'oldObs' not created (Died on first action), presume the time was 0
            try:
                results.append([score,actionCount,oldObs['TotalTime'], self.epsilon])
            except:
                results.append([score, actionCount, 0, self.epsilon])

            # Decay the epsilon until the minimum
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            else:
                # If the epsilon less than minimum, set to 0
                self.epsilon = 0
            # Store the results in the provided CSV file
            with open(self.CSVName,"w") as f:
                wr = csv.writer(f)
                wr.writerows(results)

        # Return the results
        return results

    # Function to start the game
    # Given an environment, env, and the game number, i.
    def startGame(self,env, i):
        print(" ------- New Game ----------  \n")
        # Store the Q-Table as a JSON with the provided name
        print("Saving QTable as JSON")
        with open(self.QTableName, 'w') as fp:
            json.dump(self.qTable, fp)

        # Back-Up the Q-Table every 10 games
        if (i+1) % 10 == 0:
            print("Saving QTable BackUp as JSON")
            # Store a QTable BackUp too every 10 games
            with open('QTableBackUp.json', 'w') as fp:
                json.dump(self.qTable, fp)

        # Initialise the MineCraft environment
        # Add a sleep to ensure connection to the environment
        sleep(2)
        obs = env.reset()
        # Do an initial 'stop' step in order to get info from the environment
        obs, currentReward, done, info = env.step(0)

        # Use utils module to discretise the info from the game - Find the current state
        [xdisc, ydisc, zdisc, yawdisc, pitchdisc] = utils.discretiseState(info['observation'])
        currentState = "%d:%d:%d:%d:%d" % (xdisc, zdisc, yawdisc, ydisc, pitchdisc)
        print("initialState: " + currentState)

        # Return the currentState and the first info
        return currentState, info

    # Function to choose the next action to take in the environment
    def act(self, env, currentState):

        # If no Q-Function for this state in the Q-Table, initialise it
        if currentState not in self.qTable:
            self.qTable[currentState] = ([0] * len(self.actions))

        # Select the next action

        # If the random number is less than epsilon, choose a random action
        if random.random() < self.epsilon:
            # Choose a random action from the given actions
            action = random.choice(self.actions)
            print("From State %s (X,Z,Yaw,Y,Pitch), taking random action: %s" % (currentState, action))

        # If the random number more than epsilon, choose the best action
        else:
            # Pick the highest Q-Value action for the current state
            currentStateActions = self.qTable[currentState]

            # Pick highest action Q-value - In case of tie chooses first in list
            action = self.actions[np.argmax(currentStateActions)]
            print("From State %s (X,Z,Yaw,Y,Pitch), taking q action: %s" % (currentState,  action))

        # Returns the chosen action to the game
        return action

# Main function
def main():
    # Take in command line arguments, and use for environment setup
    if len(sys.argv) > 1:
        env = utils.setupEnv(sys.argv[1])
    elif len(sys.argv) > 2:
        env = utils.setupEnv(sys.argv[1], port=sys.argv[2])
    else:
        env = utils.setupEnv()

    # Get the number of available actions
    actionSize = env.action_space.n

    # Give user decision on loading model or not
    load = input("Load Q Table? y/n - Default as y:________")

    # Set the Agent to Load Q-Table if user chooses to load
    if load.lower() == 'n':
        myAgent = QLearningAgent(actionSize,200,'QTable.json', 'qlearningResults.csv', epsilon = 1.0 )
    else:
        myAgent = QLearningAgent(actionSize,200, 'QTable.json', 'qlearningResults.csv' , True, epsilon = 1.0)

    # Start the running of the Agent
    myAgent.runAgent(env)

    return

if __name__ == "__main__":
    main()

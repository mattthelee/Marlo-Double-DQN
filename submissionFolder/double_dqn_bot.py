import marlo
import numpy as np
import random
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D,Flatten, AveragePooling2D
from collections import deque
from keras.models import model_from_yaml
from matplotlib import pyplot as plt
from past.utils import old_div # tutorial 5
import MalmoPython
import sys
import utils
import csv
from time import sleep

import pdb
from keras.backend import manual_variable_initialization


def trainAgent(env, agent):
    # Train the agent given
    # Maximum steps to take before telling agent to give up
    goal_steps = 100
    # How many games to train over
    initial_games = 10000
    # Batch for back-propagation
    batch_size = 16
    scores = deque(maxlen=50)
    results = []
    # Loop over the games initialised
    for i in range(initial_games):
        reward = 0
        game_score = 0
        # Short wait required to prevent loss of connection to marlo
        sleep(2)
        env.reset()
        state = env.last_image
        # For each step take an action and perform exprience replay
        for j in range(goal_steps):
            print("Starting goal step: ", j + 1, " of game: ", i + 1, " avg score: ", np.mean(scores))
            # Choose action
            action = agent.act(state)
            # Run action and get response from env
            new_state, reward, done, info = env.step(action)

            # Useful debug line: print(f"Taking action {action}, got reward: {reward}")
            # Adds this state, action, new state to memory
            agent.memory.append((state,action, reward, new_state, done))
            # Record gamescore for analysis
            game_score += reward
            # If game is done we break from loop and store score
            if done:
                # Score is the scores for finished games
                print("Game: ",i ," complete, score: " , game_score," last 50 scores avg: ", np.mean(scores), " epsilon ", agent.epsilon)
                scores.append(game_score)
                break

            state = new_state
            oldInfo = info

            # If we don't have enough memory for a batch, don't run experience replay
            if len(agent.memory) > batch_size:
                # Find a random batch from the memory
                randomBatch = random.sample(agent.memory, batch_size)
                # Perform experience replay
                agent.replay(randomBatch)

        # Record the stats about this game, for analysis and save to csv
        results.append([game_score,j,oldInfo['observation']['TotalTime'], agent.epsilon])
        with open(agent.CSVName,"w") as f:
            wr = csv.writer(f)
            wr.writerows(results)
        # Decay the epsilon until the minimum
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        else:
            agent.epsilon = 0
        # Save the model
        agent.saveModelToFile(agent.model,'model')
        # every 10 games update the secondary model, starting from the 3rd
        # This way the secondary model will always be at least 10 games behind the primary model
        if i == 2:
            agent.saveModelToFile(agent.model,'secondary')
            agent.secondaryDQN = agent.loadModelFromFile('secondary')
        if i % 10 == 3:
            agent.secondaryDQN = agent.loadModelFromFile('secondary')
            agent.saveModelToFile(agent.model,'secondary')
    return scores

class agent:
    def __init__(self, observation_shape, action_size, load_model_file = False, epsilon = 1.0):
        # Initialise parameters for the agent
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.block_list = ['air','cobblestone','stone','gold_block']
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95   # discount rate
        self.epsilon_min = 0.01
        self.epsilon = epsilon
        self.epsilon_decay = 0.99
        self.CSVName = 'dqn_bot_results.csv'
        if load_model_file:
            self.model = self.loadModelFromFile('model')
            self.secondaryDQN = self.loadModelFromFile('secondary')
        else:
            # Start from scratch
            self.model = self.create_model()
            self.secondaryDQN = self.create_model()

    def create_model(self):
        # Create DQN using keras Sequential api
        model = Sequential()
        # This average pooling layer is quite extreme because of memory limits on machine
        model.add(AveragePooling2D(pool_size=(8, 8), input_shape=(self.observation_shape)))
        model.add(Conv2D(32, 8, 4))
        model.add(Conv2D(16, 4, 2))
        model.add(MaxPooling2D(pool_size=(4,4)))
        # Flatten needed to get a single vector as output otherwise get a matrix
        model.add(Flatten())
        model.add(Dense(64,activation='relu'))
        model.add(Dense(64,activation='relu'))
        model.add(Dense(self.action_size,activation='linear'))
        # Other optimisers are available, such as adam
        model.compile(loss='mse', optimizer='rmsprop')
        return model

    def loadModelFromFile(self,file):
        # Loads a previous model
        # Load strucutre and weights separately to prevent tensorflow intialising and deleting weights
        yaml_file = open(file + '.yaml', 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        model = model_from_yaml(loaded_model_yaml)
        model.load_weights(file + '_weights.h5')
        model.compile(loss='mse', optimizer='rmsprop')
        return model

    def saveModelToFile(self,model,file):
        # Saves model structure and weights to file
        model_yaml = model.to_yaml()
        with open(file + ".yaml", "w") as yaml_file:
            yaml_file.write(model_yaml)
        model.save_weights(file+'_weights.h5')
        return

    def act(self, state):
        # Return the epsilon-greedy action for this state
        if np.random.rand() <= self.epsilon:
            print("Random Action")
            return random.randrange(self.action_size)
        # Reshape required because of a quirk in the Keras API
        act_values = self.model.predict(state.reshape([-1, 600, 800, 3]))
        return np.argmax(act_values[0])

    def replay(self, batch):
        # Perform experience replay using the mbatch of memories supplied
        x_train = []
        y_train = []
        for state, action, reward, newState, done in batch:
            if done or len(self.memory) < 300:
                # If finished or network has not had time to learn reasonable values
                # Set target_q to be reward
                target_q = reward
            else:
                # Use Bellman equation to calculate the q we should haves
                # N.b. This is where the double DQN differs by using the secondaryDQN not the primary
                target_q = reward + self.gamma * np.amax(self.secondaryDQN.predict(newState.reshape([-1, 600, 800, 3])))

            # prediction is prediction_q
            # prediction has the 5 actions and predicted q-values
            prediction = self.model.predict(state.reshape([-1, 600, 800, 3]))
            # Useful debug line: print(f"action: {action}, reward:{reward}, qval:{target_q}, predq:{prediction[0][action]}")

            # update the action that we did take with a better target, from above. Keep others the same to not influence the network
            prediction[0][action] = target_q

            # Create the training data for X and Y that we use to fit the DQN on
            x_train.append(state)
            y_train.append(prediction[0])

        # Use the training data to fit the model, via the batch
        self.model.fit(np.asarray(x_train),np.asarray(y_train),epochs=1,verbose=0)
        return

def main():
    # If arguments are supplied when running the agent, pass them to the setup env function, else use defaults
    if len(sys.argv) > 1:
        env = utils.setupEnv(sys.argv[1])
    elif len(sys.argv) > 2:
        env = utils.setupEnv(sys.argv[1], port=sys.argv[2])
    else:
        env = utils.setupEnv()

    #  Get the number of available states and actions - generates the output of CNN
    observation_shape = env.observation_space.shape
    action_size = env.action_space.n

    # Initialise agent and then run it.
    myagent = agent(observation_shape, action_size, False,1.0)
    scores = trainAgent(env, myagent)
    '''
    #Can start from a pre-built model
    load = input("Load model? y/n or an epsilon value to continue: ")
    if load == 'y':
        myagent = agent(observation_shape, action_size, block_map_shape,True,0.1)
        #pdb.set_trace()
        scores = testAgent(env,myagent)
    elif load == 'n':
        myagent = agent(observation_shape, action_size,block_map_shape)
        #pdb.set_trace()
        scores = trainAgent(env, myagent)
    else:
        #TODO - how come the 'epsilon value' runs still load a model??
        myagent = agent(observation_shape, action_size, block_map_shape,True,float(load))
        scores = trainAgent(env,myagent)
    '''
    np.savetxt('dqn_botscores',np.array(scores))
    #plt.plot(scores)
    #plt.show()
    return

if __name__ == "__main__":
    main()

    def blockEncoder(floorList):
        # ***This function no longer used as was planned for intepreting map data for DQN ***
        # We need to convert the block names from strings to vectors as they are categorical data
        # takes in a i-length list of the blocks with j different block types and returns an i*j length list indicating the encoded version.
        blockList = self.blockList
        # TODO need to simplfy the classes to classify these under a type of: air, goal, solid, danger (lava)
        blockDict = {}
        for i,block in enumerate(blockList):
            blockDict[block] = np.zeros(len(blockList))
            blockDict[block][i] = 1

        vectorisedList = []
        for i in floorList:
            # Adds content of list to other list. N.B. we might want to use append here depending on how we handle the data
            vectorisedList.extend(blockDict[i])
        return vectorisedList

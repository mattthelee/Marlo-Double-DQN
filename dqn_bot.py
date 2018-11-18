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

import pdb
from keras.backend import manual_variable_initialization


# Notes:
# env.action_space.sample() gives a random action from those available, applies to any env
# TODO The loaded model seems to always give action 2 as the best one. I suspsect this is n issue with the loading as the model performes better after a period of training
# I've checked that the model correctly pulls in weights so it can't be that.
# TODO to make best use of the DQN I need to use an LTSM or stack previous game frames before sending them to the NN. See the deepmind atari paper or hausknecht and stone 2015
# TODO info data contains the orientation and position of the agent, could use this as a feature to train the nn. That might be best as a separate nn that takes in the history of actions taken How does it deal with the straint position being none zero, how does it deal with the maps changing?
#   L Even just having the NN output the position it thinks its in would be a useful thing to train. What other features are valuable to extract from the images?
# TODO consider transfer learining from pretrained CNN
def trainAgent(env, agent):
    goal_steps = 500
    initial_games = 10000
    batch_size = 16
    scores = deque(maxlen=50)
    for i in range(initial_games):
        reward = 0
        game_score = 0
        env.reset()
        state = env.last_image
        for j in range(goal_steps):
            print("Starting goal step: ", j, " of game: ", i, " avg score: ", np.mean(scores))
            action = agent.act(state)
            new_state, reward, done, info = env.step(action)
            agent.memory.append((state,action, reward, new_state, done))

            if done:
                print("Game: ",i ," complete, score: " , game_score," last 50 scores avg: ", np.mean(scores), " epsilon ", agent.epsilon)
                scores.append(game_score)
                break
            game_score += reward
            state = new_state


            if len(agent.memory) > batch_size:
                randomBatch = random.sample(agent.memory, batch_size)
                agent.replay(randomBatch)
    model_yaml = agent.model.to_yaml()
    with open("model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    agent.model.save_weights('model_weights.h5')
    return scores

def testAgent(env, agent):
    goal_steps = 500
    initial_games = 50
    scores = deque(maxlen=50)
    for i in range(initial_games):
        reward = 0
        game_score = 0
        env.reset()
        state = env.last_image
        for j in range(goal_steps):
            action = agent.act(state)
            print("Starting goal step: ", j, " of game: ", i, " avg score: ", np.mean(scores), " action: ", action)
            new_state, reward, done, info = env.step(action)
            #pdb.set_trace()
            if done:
                print("Game: ",i ," complete, score: " , game_score," last 50 scores avg: ", np.mean(scores), " epsilon ", agent.epsilon)
                scores.append(game_score)
                break
            game_score += reward
            state = new_state
    return scores

class agent:
    def __init__(self, observation_shape, action_size, load_model_file = False, epsilon = 1.0):
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon_min = 0.01
        self.epsilon = epsilon
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        if load_model_file:
            # This is required to stop tensorflow reinitialising weights on model load
            #manual_variable_initialization(True)
            #self.model = load_model('model.h5')
            #self.model.load_weights('model.h5')
            yaml_file = open('model.yaml', 'r')
            loaded_model_yaml = yaml_file.read()
            yaml_file.close()
            self.model = model_from_yaml(loaded_model_yaml)
            self.model.load_weights('model_weights.h5')
        else:
            self.model = self.create_model()

    def create_model(self):
        model = Sequential()
        # Need to check that this is processing the colour bands correctly <- have checked this and:
        # the default is channels last which is what we have
        # This max pooling layer is quite extreme because of memory limits on machine
        model.add(AveragePooling2D(pool_size=(8, 8), input_shape=(self.observation_shape)))
        model.add(Conv2D(32, 8, 4)) # Convolutions set to same as in Lample and Chaplet
        model.add(Conv2D(64, 4, 2)) # Convolutions set to same as in Lample and Chaplet

        # Flatten needed to get a single vector as output otherwise get a matrix
        model.add(Flatten())
        model.add(Dense(128,activation='relu'))
        model.add(Dense(64,activation='relu'))
        model.add(Dense(self.action_size,activation='linear'))
        model.compile(loss='mse', optimizer='rmsprop')
        return model

    def act(self, state):
        # Randomly choose to take a randomly chosen action to allow exploration
        # When epsilon is high, higher chance, therefore decrease it overtime
        # This then results in exploration early on with greater exploitation later
        if np.random.rand() <= self.epsilon:
            print("Random Action")
            return random.randrange(self.action_size)
        act_values = self.model.predict(state.reshape([-1, 600, 800, 3]))
        return np.argmax(act_values[0])

    def replay(self, batch):
        x_train = []
        y_train = []
        for state, action, reward, newState, done in batch:
            if done:
                # Set the reward for finishing the game
                target_q = reward
            else:
                #pdb.set_trace()
                #self.model.predict(newState.reshape([-1, 600, 800, 3]))
                target_q = reward + self.gamma * np.amax(self.model.predict(newState.reshape([-1, 600, 800, 3])))
            prediction = self.model.predict(state.reshape([-1, 600, 800, 3]))
            prediction[0][action] = target_q
            x_train.append(state)
            y_train.append(prediction[0])
        self.model.fit(np.asarray(x_train),np.asarray(y_train),epochs=1,verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = 0
        return

def Menger(xorg, yorg, zorg, size, blocktype, variant, holetype):
    # Needed to run the tutorial 5 mission xml
    #draw solid chunk
    genstring = GenCuboidWithVariant(xorg,yorg,zorg,xorg+size-1,yorg+size-1,zorg+size-1,blocktype,variant) + "\n"
    #now remove holes
    unit = size
    while (unit >= 3):
        w=old_div(unit,3)
        for i in range(0, size, unit):
            for j in range(0, size, unit):
                x=xorg+i
                y=yorg+j
                genstring += GenCuboid(x+w,y+w,zorg,(x+2*w)-1,(y+2*w)-1,zorg+size-1,holetype) + "\n"
                y=yorg+i
                z=zorg+j
                genstring += GenCuboid(xorg,y+w,z+w,xorg+size-1, (y+2*w)-1,(z+2*w)-1,holetype) + "\n"
                genstring += GenCuboid(x+w,yorg,z+w,(x+2*w)-1,yorg+size-1,(z+2*w)-1,holetype) + "\n"
        unit = w
    return genstring

def GenCuboid(x1, y1, z1, x2, y2, z2, blocktype):
    return '<DrawCuboid x1="' + str(x1) + '" y1="' + str(y1) + '" z1="' + str(z1) + '" x2="' + str(x2) + '" y2="' + str(y2) + '" z2="' + str(z2) + '" type="' + blocktype + '"/>'

def GenCuboidWithVariant(x1, y1, z1, x2, y2, z2, blocktype, variant):
    return '<DrawCuboid x1="' + str(x1) + '" y1="' + str(y1) + '" z1="' + str(z1) + '" x2="' + str(x2) + '" y2="' + str(y2) + '" z2="' + str(z2) + '" type="' + blocktype + '" variant="' + variant + '"/>'

def loadMissionFile(filename):
    with open(filename, 'r') as file:
        missionXML = file.read()
    return missionXML


def main():
    client_pool = [('127.0.0.1', 10000)]
    # Suppress info set to false to allow the agent to train using additional features, this will be off for testing!
    join_tokens = marlo.make('MarLo-FindTheGoal-v0', params={"client_pool": client_pool, 'suppress_info': False})
    # As this is a single agent scenario,
    # there will just be a single token
    assert len(join_tokens) == 1
    join_token = join_tokens[0]

    env = marlo.init(join_token)
    # Change the spec of the mission by loading xml from file
    missionXML= loadMissionFile('find_the_goal_mission2.xml')
    #missionXML= loadMissionFile('mission_spec')

    env.mission_spec = MalmoPython.MissionSpec(missionXML, True)

    # Get the number of available states and actions
    observation_shape = env.observation_space.shape
    action_size = env.action_space.n
    #pdb.set_trace()
    load = input("Load model? y/n or an epsilon value to continue: ")

    if load == 'y':
        myagent = agent(observation_shape, action_size,True,0.1)
        #pdb.set_trace()
        scores = testAgent(env,myagent)
    elif load == 'n':
        myagent = agent(observation_shape, action_size)
        pdb.set_trace()
        scores = trainAgent(env, myagent)
    else:
        myagent = agent(observation_shape, action_size,True,float(load))
        scores = trainAgent(env,myagent)

    print (scores)
    np.savetxt('scores',np.array(scores))
    #plt.plot(scores)
    #plt.show()
    return

if __name__ == "__main__":
    main()

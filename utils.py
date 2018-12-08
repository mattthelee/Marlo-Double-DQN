from marlo import MalmoPython
import json
import marlo
from numpy import genfromtxt
from matplotlib import pyplot as plt
import pdb
from time import sleep

def discretiseState(obs,toNearest = [0.5,45]):
    # Takes in info dictionary and returns the x,y,z,yaw,pitch discretised as list
    x = obs['XPos']
    y = obs['YPos']
    z = obs['ZPos']
    yaw = obs['Yaw']
    pitch = obs['Pitch']
    # Round to nearest[0]
    xdisc = round(x / toNearest[0])*toNearest[0]
    ydisc = round(y / toNearest[0])*toNearest[0]
    zdisc = round(z / toNearest[0])*toNearest[0]
    # Round to nearest 45
    yawdisc = round(yaw / toNearest[1])*toNearest[1]
    # Ensure the yaw is a positive output
    if yawdisc == 360:
        yawdisc = 0
    if yawdisc < 0:
        yawdisc = 360 + yawdisc
    pitchdisc = round(pitch/ toNearest[1])*toNearest[1]
    return [xdisc,ydisc,zdisc,yawdisc,pitchdisc]

def loadMissionFile(filename):
    with open(filename, 'r') as file:
        missionXML = file.read()
    return missionXML

def setupEnv(mission='MarLo-FindTheGoal-v0', videoResolution = [800, 600]):
    client_pool = [('127.0.0.1', 10000)]
    # Step sleep at to 0.2 to handle lag between marlo and Malmo
    join_tokens = marlo.make(mission, params={
        "client_pool": client_pool,
        'suppress_info': False,
        'videoResolution': videoResolution,
        'tick_length': 50,
        'step_sleep': 0.2})
    # As this is a single agent scenario,
    # there will just be a single token
    assert len(join_tokens) == 1
    join_token = join_tokens[0]
    env = marlo.init(join_token)
    # Change the spec of the mission by loading xml from file
    missionXML= loadMissionFile(mission+'.xml')
    env.mission_spec = MalmoPython.MissionSpec(missionXML, True)
    return env


def rotateSeenBlocks(floor11x11x3, yaw):
    # Presuming this blockInput to be 3 layers of 11x11 - in np.array as (3,11,11)

    # Set a threshold, T, for the difference of yaw to the distinct 90 degrees
    T = 15

    # Initialise the blocks variable as a np.array (3,11,11)
    blocks = np.zeros((3, 11, 11))

    # Loop through the 3 heights in floor11x11x3
    for i, level11x11 in enumerate(floor11x11x3):
        # If at yaw = 0
        if abs(yaw) < T:
            # No rotation required
            blocks[i] = level11x11
        # If at yaw = 90
        if abs(yaw - 90) < T:
            # Rotate counter-clockwise 90
            blocks[i] = np.rot90(level11x11, 1)
        # If at yaw = 180
        if abs(yaw - 180) < T:
            # Rotate counter-clockwise 180
            blocks[i] = np.rot90(level11x11, 2)
        # If at yaw = 270
        if abs(yaw - 270) < T:
            # Rotate clockwise 90
            blocks[i] = np.rot90(level11x11, -1)

    # Generate the output block of 4x5 for each
    seenBlocks3x4x5 = blocks[:, 0:4, 3:8]

    return seenBlocks3x4x5

def plotResults(file):
    my_data = genfromtxt(file, delimiter=',')
    y = [x[0] for x in my_data]
    plt.plot(y)
    input('waiting')
    return
'''
new_state, reward, done, info = env.step(0)
print(discretiseState(json.loads(env._get_world_state().observations[-1].text)))
new_state, reward, done, info = discreteMove(env,1, info)
print(discretiseState(json.loads(env._get_world_state().observations[-1].text)))


new_state, reward, done, info = env.step(0)
print(utils.discretiseState(json.loads(env._get_world_state().observations[-1].text)))
new_state, reward, done, info = utils.discreteMove(env,1, info)
print(utils.discretiseState(json.loads(env._get_world_state().observations[-1].text)))
'''

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
    # Round to whatever is set by  nearest[0] defaults to 0.5
    xdisc = round(x / toNearest[0])*toNearest[0]
    ydisc = round(y / toNearest[0])*toNearest[0]
    zdisc = round(z / toNearest[0])*toNearest[0]
    # Round to whatever is set by nearest[1] defaults to 45
    yawdisc = round(yaw / toNearest[1])*toNearest[1]
    # Ensure the yaw is a positive output
    if yawdisc == 360:
        yawdisc = 0
    if yawdisc < 0:
        yawdisc = 360 + yawdisc
    pitchdisc = round(pitch/ toNearest[1])*toNearest[1]
    return [xdisc,ydisc,zdisc,yawdisc,pitchdisc]

def actionCompleted(obs1,obs2,action):
    # If action is not a movement return false
    if action in [0,5,6]:
        return False
    # If discretised position or angle has changed return true
    if abs(max([a -b for a,b in zip(discretiseState(obs1),discretiseState(obs2))], key=abs
)) > 0:
        return True
    return False


def loadMissionFile(filename):
    # Loads XML mission files
    with open(filename, 'r') as file:
        missionXML = file.read()
    return missionXML

def setupEnv(mission='MarLo-FindTheGoal-v0', videoResolution = [800, 600], port=10000):
    # Sets up marlo environment
    client_pool = [('127.0.0.1', port)]
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
    # This is necessary to use our custom XML files
    missionXML= loadMissionFile(mission+'.xml')
    env.mission_spec = MalmoPython.MissionSpec(missionXML, True)
    return env


def plotResults(file):
    # Simple plotting function
    my_data = genfromtxt(file, delimiter=',')
    y = [x[0] for x in my_data]
    plt.plot(y)
    input('waiting')
    return

def rotateSeenBlocks(floor11x11x3, yaw):
    # *** Not used in final implementation as DQN does not use block map information as part of DQN training
    # Rotates blocks to match the perspective the DQN will receive images in, so it only considers visble blocks
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

def completeAction(env,action):
    # *** Not used in final implementation as discrete state combined with step_sleep resilved issues
    # This function solves the 2 problems with the env.step method by circumventing the Marlo API and hitting Malmo directly
    # 1) Actions do not always take effect immediately, therefore do an action and wait for state change before returning
    # 2) Marlo does not provide the reward for the action just taken but for the previous action, need to do wait action before
    env._take_action(0)
    sleep(0.1)
    env._take_action(action)
    # This sleep is required to let marlo 'settle' into its state. The state is then taken from the world state object
    sleep(0.2)
    reward = 0
    world_state = env._get_world_state()
    for _reward in world_state.rewards:
        reward += _reward.getValue()
    # Tries to return the state by queying world state, it will fail if gameover though
    # in which case it should do the action again to get the final reward
    # Get observation
    image = env._get_video_frame(world_state)

    # detect if done ?
    done = not world_state.is_mission_running or any(world_state.errors)

    # Notify evaluation system, if applicable
    # marlo.CrowdAiNotifier._env_action(action)
    marlo.CrowdAiNotifier._step_reward(reward)
    if done:
        marlo.CrowdAiNotifier._episode_done()
    try:
        return image, reward, done, json.loads(world_state.observations[-1].text)
    except:
        image, reward3, done, info = env.step(action)
        reward += reward3
        return image, reward, done, info['observation']

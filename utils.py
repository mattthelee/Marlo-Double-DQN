def discreteMove(env, action):
    # Number of repetitions of each action to perform.
    actionReps = [1,2,2,5,5,1,3,2,2]
    totalReward = 0
    for i in range(actionReps[action]):
        new_state, reward, done, info = env.step(action)
        totalReward += reward
        # Stop doing the action
        new_state, reward, done, info = env.step(0)
        totalReward += reward
        print('Doing action:',action)
        if done:
            break
    return new_state, totalReward, done, info

def discretiseState(info):
    # Takes in info dictionary and returns the x,y,z,yaw,pitch discretised as list
    x = info['observation']['XPos']
    y = info['observation']['YPos']
    z = info['observation']['ZPos']
    yaw = info['observation']['Yaw']
    pitch = info['observation']['Pitch']

    # Round to nearest 0.5
    xdisc = round(x * 2)/2
    ydisc = round(y * 2)/2
    zdisc = round(z * 2)/2

    # Round to nearest 45
    yawdisc = round(yaw / 45)*45
    pitchdisc = round(pitch/ 45)*45

    return [xdisc,ydisc,zdisc,yawdisc,pitchdisc]





x1 = info['observation']['XPos']
y1 = info['observation']['YPos']
z1 = info['observation']['ZPos']
yaw1 = info['observation']['Yaw']
pitch1 = info['observation']['Pitch']
new_state, reward, done, info = discreteMove(env,1)
new_state, reward, done, info = discreteMove(env,2)
print(info['observation']['XPos']-x1)
print(info['observation']['YPos']-y1)
print(info['observation']['ZPos']-z1)
print(info['observation']['Yaw']-yaw1)
print(info['observation']['Pitch']-pitch1)

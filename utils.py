def discreteMove(env, action, info):
    # Max Number of repetitions of each action to perform.
    actionReps = [1,10,10,10,10,1,1,10,10]
    totalReward = 0
    info1 = info
    for i in range(actionReps[action]):
        new_state, reward, done, info = env.step(action)
        totalReward += reward
        # Stop doing the action
        new_state, reward, done, info = env.step(0)
        totalReward += reward
        if done or actionCompleted(info1,info,action):
            break
    return new_state, totalReward, done, info

def discretiseState(info,toNearest = [0.5,45]):
    # Takes in info dictionary and returns the x,y,z,yaw,pitch discretised as list
    x = info['observation']['XPos']
    y = info['observation']['YPos']
    z = info['observation']['ZPos']
    yaw = info['observation']['Yaw']
    pitch = info['observation']['Pitch']
    # Round to nearest[0]
    xdisc = round(x / toNearest[0])*toNearest[0]
    ydisc = round(y / toNearest[0])*toNearest[0]
    zdisc = round(z / toNearest[0])*toNearest[0]
    # Round to nearest 45
    yawdisc = round(yaw / toNearest[1])*toNearest[1]
    pitchdisc = round(pitch/ toNearest[1])*toNearest[1]
    return [xdisc,ydisc,zdisc,yawdisc,pitchdisc]

def actionCompleted(info1,info2,action):
    # If action is not a movement return false
    if action in [0,5,6]:
        return False
    # If discretised position or angle has changed return true
    if abs(max([a -b for a,b in zip(discretiseState(info1),discretiseState(info2))], key=abs
)) > 0:
        return True
    return False

new_state, reward, done, info = env.step(0)
print(discretiseState(info))
new_state, reward, done, info = discreteMove(env,1, info)
print(discretiseState(info))

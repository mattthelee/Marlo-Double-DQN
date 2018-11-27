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

def discretiseState(state):
    


x1 = info['observation']['XPos']
y1 = info['observation']['YPos']
z1 = info['observation']['ZPos']
yaw1 = info['observation']['Yaw']
pitch1 = info['observation']['Pitch']
new_state, reward, done, info = discreteMove(env,1)
print(info['observation']['XPos']-x1)
print(info['observation']['YPos']-y1)
print(info['observation']['ZPos']-z1)
print(info['observation']['Yaw']-yaw1)
print(info['observation']['Pitch']-pitch1)

def completeAction(env,action):
    # Actions do not always take effect immediately, therefore do an action and wait for state change before returning
    # Because Marlo does not provide the reward for the action just taken but for the previous action, need to do wait action before
    image, reward, done, info = env.step(0)
    image, reward, done, info = env.step(action)
    if done:
        return image, reward, done, info['observation']
    # Check ten times if action has been completed
    obs = info['observation']
    for i in range(10):
        # If the game is over the observation history is 0
        if len(env._get_world_state().observations) == 0:
            done = True
            break
        obs = json.loads(env._get_world_state().observations[-1].text)
        if actionCompleted(info['observation'],obs,action):
            break
    return image, reward, done, obs
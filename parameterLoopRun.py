from QLearning import QLearningAgent
import utils

def main():
    env = utils.setupEnv('find_the_goal_mission2.xml')
    # Get the number of available actions, minus waiting action
    actionSize = env.action_space.n

    epsilonDecay = 0.97
    alphas = [0.8]
    gammas = [1]

    #alphas = [0.01,0.1,0.8]
    #gammas = [0.5,1]

    for alpha in alphas:
        for gamma in gammas:
            QTableName = "QTable_Alpha_" + str(alpha).replace(".", "_") + "_Gamma_" + str(gamma).replace(".","_") + "_Decay_" + str(epsilonDecay).replace(".", "_") + ".json"
            CSVName = "Results_Alpha_" + str(alpha).replace(".", "_") + "_Gamma_" + str(gamma).replace(".", "_")+ "_Decay_" + str(epsilonDecay).replace(".", "_") + ".csv"

            myAgent = QLearningAgent(actionSize,250, QTableName,CSVName, False, epsilonDecay , alpha, gamma)

            # Start the running of the Agent
            myAgent.runAgent(env)

    return

if __name__ == "__main__":
    main()

from QLearning import QLearningAgent
import utils

def main():
    env = utils.setupEnv('MarLo-TrickyArena-v0')
    # Get the number of available actions, minus waiting action
    actionSize = env.action_space.n

    epsilonDecay = 0.98
    alphas = [0.8,0.5,0.1]
    gammas = [1,0.5]


    for alpha in alphas:
        for gamma in gammas:

            QTableName = "QTable_Alpha_" + str(alpha).replace(".", "_") + "_Gamma_" + str(gamma).replace(".","_") + "_Decay_" + str(epsilonDecay).replace(".", "_") + ".json"
            CSVName = "Results_Alpha_" + str(alpha).replace(".", "_") + "_Gamma_" + str(gamma).replace(".", "_")+ "_Decay_" + str(epsilonDecay).replace(".", "_") + ".csv"

            myAgent = QLearningAgent(actionSize,200, QTableName,CSVName, False, epsilonDecay , alpha, gamma)

            # Start the running of the Agent
            myAgent.runAgent(env)

        return

if __name__ == "__main__":
    main()

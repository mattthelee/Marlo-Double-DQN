from QLearning import QLearningAgent
import utils

def main():
    env = utils.setupEnv('MarLo-Vertical-v0')
    # Get the number of available actions, minus waiting action
    #actionSize = env.action_space.n
    actionSize = 5

    epsilonDecay = 0.97
    #alphas = [0.8,0.5,0.1]
    #gammas = [1,0.5]

    alphas = [0.5]
    gammas = [1]
    i = 1

    for alpha in alphas:
        for gamma in gammas:
            QTableName = "QTable_Alpha_" + str(alpha).replace(".", "_") + "_Gamma_" + str(gamma).replace(".","_") + "_Decay_" + str(epsilonDecay).replace(".", "_") + ".json"
            CSVName = str(i) + "_Test_Results_Alpha_" + str(alpha).replace(".", "_") + "_Gamma_" + str(gamma).replace(".", "_")+ "_Decay_" + str(epsilonDecay).replace(".", "_") + ".csv"

            myAgent = QLearningAgent(actionSize, 25, QTableName,CSVName, True, epsilonDecay , alpha, gamma,0.00,training = True)

            print("\n\n -------------- Starting test run of Decay %s, Alpha %s and Gamma %s --------- \n \n" % (epsilonDecay,alpha,gamma))

            # Start the running of the Agent
            myAgent.runAgent(env)

    return

if __name__ == "__main__":
    main()
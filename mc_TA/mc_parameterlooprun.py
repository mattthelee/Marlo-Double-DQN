
from mc import MC_agent
import utils


def main():
    env = utils.setupEnv("MarLo-FindTheGoal-v0")
    # Get the number of available actions, minus waiting action
    actionSize = env.action_space.n

    epsilonDecay = 0.97
    #not alpha
    #have 6 values for gamma
    # which ones are best then why we used values for the q-learning
    #best two run for cliff and and treck a
    gammas = [1,0.8,0.6,0.4,0.2,0]


    for gamma in gammas:
        mc_QTableName = "mc_QTable_Gamma_" + str(gamma).replace(".","_") + "_Decay_" + str(epsilonDecay).replace(".", "_") + ".json"
        mc_CSVName = "mc_Results_Gamma_" + str(gamma).replace(".", "_")+ "_Decay_" + str(epsilonDecay).replace(".", "_") + ".csv"

        myAgent = MC_agent(actionSize, mc_QTableName,mc_CSVName, False, epsilonDecay , gamma)

            # Start the running of the Agent
        myAgent.runAgent(env)

    return

if __name__ == "__main__":
    main()

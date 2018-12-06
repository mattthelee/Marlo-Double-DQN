
from mc_loops.mc import MC_agent
import mc_loops.utils

def main():
    env = mc_loops.utils.setupEnv('find_the_goal_mission2.xml')
    # Get the number of available actions, minus waiting action
    actionSize = env.action_space.n

    epsilonDecay = 0.99
    alphas = [0.01,0.08]
    gammas = [1,0.5]

    for alpha in alphas:
        for gamma in gammas:
            mc_QTableName = "mc_QTable_Alpha_" + str(alpha).replace(".", "_") + "_Gamma_" + str(gamma).replace(".","_") + "_Decay_" + str(epsilonDecay).replace(".", "_") + ".json"
            mc_CSVName = "mc_Results_Alpha_" + str(alpha).replace(".", "_") + "_Gamma_" + str(gamma).replace(".", "_")+ "_Decay_" + str(epsilonDecay).replace(".", "_") + ".csv"

            myAgent = MC_agent(actionSize, mc_QTableName,mc_CSVName, False, epsilonDecay , alpha, gamma)

            # Start the running of the Agent
            myAgent.runAgent(env)

    return

if __name__ == "__main__":
    main()

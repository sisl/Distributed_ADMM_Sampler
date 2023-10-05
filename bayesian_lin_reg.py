"""
    In this file, we regenerate the results of
        "Decentralized Stochastic Gradient Langevin Dynamics and Hamiltonian Monte Carlo"
    in the case of Bayesian linear regression. 

    We implement D-ULA from
        "A Decentralized Approach to Bayesian Learning".

    Further, we implement our ADMM-based sampling algorithm.
"""

#######################################################
# Add dependencies
import numpy as np
import cvxpy as cp
import networkx as nx
import numpy.random as random
import matplotlib.pyplot as plt
from sklearn.covariance import EmpiricalCovariance
import scipy as sp
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings("ignore")

#######################################################
###########     PARAMETER DEFINITION        ###########

# Set random seed
random.seed(10)

# Define dimensionality of state space
dim = 2

# Define network connectivity
num_agents = 20
graph = nx.complete_graph(num_agents)

# Define number of samples per agent
num_samples_per_agent = 50

# Data generation
ksi = 4.
lamda = 10

X_per_agent = []
y_per_agent = []

x = random.multivariate_normal(np.zeros(dim), lamda*np.eye(dim) )

for aa in range(num_agents):
    X_agent = random.multivariate_normal(np.zeros(dim), np.eye(dim), size=(num_samples_per_agent,)).T

    X_per_agent.append(X_agent)
    y_per_agent.append(x.T@X_agent + random.normal(0, ksi, size=(num_samples_per_agent,)))

X = np.hstack(X_per_agent)
y = np.hstack(y_per_agent)

# print(X_per_agent[0].shape) # (2,50)
# print(y_per_agent[0].shape) # (1,50)

# Determine posterior moments
mu_posterior = np.linalg.inv( (1/lamda)*np.eye(dim) + X@X.T/ksi**2 ) @ X@y.T/ksi**2
cov_posterior = np.linalg.inv( (1/lamda)*np.eye(dim) + X@X.T/ksi**2 )

# Define gradients
grad_fi = lambda _x, _X, _y: np.sum(-(_y-_x.T@_X)*_X/ksi**2, axis=1) + _x/(lamda*num_agents)

# Define step size 
eta_dsgld = 0.009
eta_dsghmc = 0.1

gamma = 7.

alpha_0 = 0.00082
beta_0  = 0.48
b_1 = 230
b_2 = 230
delta_1 = 0.55
delta_2 =  0.05

# Define number of iterations
num_iterations = 50 

# Define number of trials
num_trials = 100 

# Include all parameters in dictionary
all_params_and_data = {}
all_params_and_data['graph'] = graph
all_params_and_data['dim'] = dim
all_params_and_data['num_agents'] = num_agents
all_params_and_data['mu_posterior'] = mu_posterior
all_params_and_data['cov_posterior'] = cov_posterior
all_params_and_data['eta_dsgld'] = eta_dsgld
all_params_and_data['eta_dsghmc'] = eta_dsghmc
all_params_and_data['gamma'] = gamma
all_params_and_data["alpha_0"] = alpha_0
all_params_and_data["beta_0"] = beta_0
all_params_and_data["b_1"] = b_1
all_params_and_data["b_2"] = b_2
all_params_and_data["delta_1"] = delta_1
all_params_and_data["delta_2"] = delta_2
all_params_and_data['num_trials'] = num_trials
all_params_and_data['num_iterations'] = num_iterations
all_params_and_data['X_per_agent'] = X_per_agent
all_params_and_data['y_per_agent'] = y_per_agent

#######################################################
###########          DEPLOY METHODS         ###########

def deploy_dynamics(method, all_params_and_data):

    # Define variables
    graph=all_params_and_data['graph']
    dim=all_params_and_data['dim']
    num_agents=all_params_and_data['num_agents']
    mu_posterior=all_params_and_data['mu_posterior']
    cov_posterior=all_params_and_data['cov_posterior']
    eta_dsgld=all_params_and_data['eta_dsgld']
    eta_dsghmc=all_params_and_data['eta_dsghmc']
    gamma=all_params_and_data['gamma']
    alpha_0=all_params_and_data["alpha_0"]
    beta_0=all_params_and_data["beta_0"]
    b_1=all_params_and_data["b_1"] 
    b_2=all_params_and_data["b_2"]
    delta_1=all_params_and_data["delta_1"]
    delta_2=all_params_and_data["delta_2"]
    num_trials=all_params_and_data['num_trials']
    num_iterations=all_params_and_data['num_iterations']
    X_per_agent=all_params_and_data['X_per_agent']
    y_per_agent=all_params_and_data['y_per_agent']

    # Define doubly stochastic matrix based on connectivity
    W = np.zeros((num_agents, num_agents))

    for edge in list(graph.edges):
        W[edge[0], edge[1]] = 1/(np.maximum(graph.degree[edge[0]], graph.degree[edge[1]])+1)
        W[edge[1], edge[0]] = W[edge[0], edge[1]]
    for aa in range(num_agents):
        W[aa, aa] = 1 - np.sum(W[aa, :])

    W_ext = np.kron(W, np.eye(dim))

    # Define output variable
    wasserstein_dists = [[] for aa in range(num_agents+1)] # last list refers to average iterate

    # Initialize trial samples
    xs_per_agent = [np.random.multivariate_normal(np.zeros((dim,)), np.eye(dim), size=(num_trials,)) for aa in range(num_agents)]

    # Different initialization
    # A = np.random.rand(dim, dim)
    # B = np.dot(A, A.transpose())
    # xs_per_agent = [np.random.multivariate_normal(np.array([1.,2.]), B, size=(num_trials,)) for aa in range(num_agents)]


    # Choose sampling method
    if method == "D-SGLD":

        eta = eta_dsgld

        # Main Loop
        for iter in tqdm(range(num_iterations)):
            ########################################################################################################################
            # Compute Wasserstein distance of agents' iterates with target
            for aa in range(num_agents):
                empirical_mean = np.mean(xs_per_agent[aa], axis=0)
                empirical_cov  = EmpiricalCovariance(assume_centered=False).fit(xs_per_agent[aa]).covariance_

                dist = np.linalg.norm(empirical_mean - mu_posterior)**2 + \
                        np.trace(empirical_cov + cov_posterior - 2*sp.linalg.sqrtm( sp.linalg.sqrtm(cov_posterior)@empirical_cov@sp.linalg.sqrtm(cov_posterior)  ) )
                
                dist = np.sqrt(dist)
                wasserstein_dists[aa].append(dist)

            # Compute Wasserstein distance of average iterate
            average_iterates = []
            for trial in range(num_trials):
                trial_iterates = [xs_per_agent[aa][trial, :] for aa in range(num_agents)]
                average_iterate = np.mean(trial_iterates, axis=0)
                average_iterates.append(average_iterate)

            empirical_mean = np.mean(average_iterates, axis=0)
            empirical_cov  = EmpiricalCovariance(assume_centered=False).fit(average_iterates).covariance_

            dist = np.linalg.norm(empirical_mean - mu_posterior)**2 + \
                    np.trace(empirical_cov + cov_posterior - 2*sp.linalg.sqrtm( sp.linalg.sqrtm(cov_posterior)@empirical_cov@sp.linalg.sqrtm(cov_posterior)  ) )
            
            dist = np.sqrt(dist)
            wasserstein_dists[-1].append(dist)


            ########################################################################################################################
            # Check agreement between agents
            # empirical_mean_1 = np.mean(xs_per_agent[0], axis=0)
            # empirical_mean_2 = np.mean(xs_per_agent[1], axis=0)
            # empirical_cov_1  = EmpiricalCovariance(assume_centered=False).fit(xs_per_agent[0]).covariance_
            # empirical_cov_2  = EmpiricalCovariance(assume_centered=False).fit(xs_per_agent[1]).covariance_

            # dist = np.linalg.norm(empirical_mean_1 - empirical_mean_2)**2 + \
            #         np.trace(empirical_cov_1 + empirical_cov_2 - 2*sp.linalg.cholesky( sp.linalg.cholesky(empirical_cov_1)@empirical_cov_2@sp.linalg.cholesky(empirical_cov_1)  ) )
            
            # dist = np.sqrt(dist)
            # wasserstein_dists[0].append(dist)

            ########################################################################################################################

            # # Plot samples at end of deployment
            # plt.figure()                        
            # for aa in range(num_agents):
            #     colors = ['b', 'g', 'r', 'm', 'k']
            #     plt.plot(xs_per_agent[aa][:, 0], xs_per_agent[aa][:, 1], '*', color = colors[aa%len(colors)])
            # plt.savefig('./Results/Bay_Lin_Reg/DeSGLD/desgld_'+str(iter))      

            ########################################################################################################################
            # Main loop
            for trial in range(num_trials):

                x_per_agent = [xs_per_agent[aa][trial, :] for aa in range(num_agents)]

                grads = [grad_fi(x_per_agent[aa], X_per_agent[aa], y_per_agent[aa]) for aa in range(num_agents)]

                for aa in range(num_agents):
                    averaging_term = np.array([W[aa, aaa]*x_per_agent[aaa] for aaa in range(num_agents)])
                    xs_per_agent[aa][trial, :] = np.sum(averaging_term, axis=0) \
                                        - eta*grads[aa] + np.sqrt(eta*2) * np.random.multivariate_normal(np.zeros(dim,), np.eye(dim))
 

    elif method == "D-ULA":

        # Main Loop
        for iter in tqdm(range(num_iterations)):
            ########################################################################################################################
            # Compute Wasserstein distance of agents' iterates with target
            for aa in range(num_agents):
                empirical_mean = np.mean(xs_per_agent[aa], axis=0)
                empirical_cov  = EmpiricalCovariance(assume_centered=False).fit(xs_per_agent[aa]).covariance_

                dist = np.linalg.norm(empirical_mean - mu_posterior)**2 + \
                        np.trace(empirical_cov + cov_posterior - 2*sp.linalg.sqrtm( sp.linalg.sqrtm(cov_posterior)@empirical_cov@sp.linalg.sqrtm(cov_posterior)  ) )
                
                dist = np.sqrt(dist)
                wasserstein_dists[aa].append(dist)

            # Compute Wasserstein distance of average iterate
            average_iterates = []
            for trial in range(num_trials):
                trial_iterates = [xs_per_agent[aa][trial, :] for aa in range(num_agents)]
                average_iterate = np.mean(trial_iterates, axis=0)
                average_iterates.append(average_iterate)

            empirical_mean = np.mean(average_iterates, axis=0)
            empirical_cov  = EmpiricalCovariance(assume_centered=False).fit(average_iterates).covariance_

            dist = np.linalg.norm(empirical_mean - mu_posterior)**2 + \
                    np.trace(empirical_cov + cov_posterior - 2*sp.linalg.sqrtm( sp.linalg.sqrtm(cov_posterior)@empirical_cov@sp.linalg.sqrtm(cov_posterior)  ) )
            
            dist = np.sqrt(dist)
            wasserstein_dists[-1].append(dist)


            ########################################################################################################################
            # Check agreement between agents
            # empirical_mean_1 = np.mean(xs_per_agent[0], axis=0)
            # empirical_mean_2 = np.mean(xs_per_agent[1], axis=0)
            # empirical_cov_1  = EmpiricalCovariance(assume_centered=False).fit(xs_per_agent[0]).covariance_
            # empirical_cov_2  = EmpiricalCovariance(assume_centered=False).fit(xs_per_agent[1]).covariance_

            # dist = np.linalg.norm(empirical_mean_1 - empirical_mean_2)**2 + \
            #         np.trace(empirical_cov_1 + empirical_cov_2 - 2*sp.linalg.cholesky( sp.linalg.cholesky(empirical_cov_1)@empirical_cov_2@sp.linalg.cholesky(empirical_cov_1)  ) )
            
            # dist = np.sqrt(dist)
            # wasserstein_dists[0].append(dist)

            ########################################################################################################################

            # # Plot samples at end of deployment
            # plt.figure()                        
            # for aa in range(num_agents):
            #     colors = ['b', 'g', 'r', 'm', 'k']
            #     plt.plot(xs_per_agent[aa][:, 0], xs_per_agent[aa][:, 1], '*', color = colors[aa%len(colors)])
            # plt.savefig('./Results/Bay_Lin_Reg/DeSGLD/desgld_'+str(iter))      

            ########################################################################################################################
            # Main loop
            alpha_k = alpha_0/(b_1+iter)**delta_2

            beta_k = beta_0/(b_2+iter)**delta_1

            for trial in range(num_trials):

                x_per_agent = [xs_per_agent[aa][trial, :] for aa in range(num_agents)]

                grads = [grad_fi(x_per_agent[aa], X_per_agent[aa], y_per_agent[aa]) for aa in range(num_agents)]

                for aa in range(num_agents):
                    averaging_term = np.array([x_per_agent[aa]- x_per_agent[aaa] for aaa in list(graph.adj[aa])])
                    xs_per_agent[aa][trial, :] = x_per_agent[aa] - beta_k*np.sum(averaging_term, axis=0) \
                                        - alpha_k*num_agents*grads[aa] + np.sqrt(alpha_k*2) * np.random.multivariate_normal(np.zeros(dim,), num_agents*np.eye(dim))
 

    elif method == "D-SGHMC":

        eta = eta_dsghmc

        vs_per_agent = [np.random.multivariate_normal(np.zeros((dim,)), np.eye(dim), size=(num_trials,)) for aa in range(num_agents)]

        # Main Loop
        for iter in tqdm(range(num_iterations)):
            ########################################################################################################################
            # Compute Wasserstein distance of agents' iterates with target
            for aa in range(num_agents):
                empirical_mean = np.mean(xs_per_agent[aa], axis=0)
                empirical_cov  = EmpiricalCovariance(assume_centered=False).fit(xs_per_agent[aa]).covariance_

                dist = np.linalg.norm(empirical_mean - mu_posterior)**2 + \
                        np.trace(empirical_cov + cov_posterior - 2*sp.linalg.sqrtm( sp.linalg.sqrtm(cov_posterior)@empirical_cov@sp.linalg.sqrtm(cov_posterior)  ) )
                
                dist = np.sqrt(dist)
                wasserstein_dists[aa].append(dist)

            # Compute Wasserstein distance of average iterate
            average_iterates = []
            for trial in range(num_trials):
                trial_iterates = [xs_per_agent[aa][trial, :] for aa in range(num_agents)]
                average_iterate = np.mean(trial_iterates, axis=0)
                average_iterates.append(average_iterate)

            empirical_mean = np.mean(average_iterates, axis=0)
            empirical_cov  = EmpiricalCovariance(assume_centered=False).fit(average_iterates).covariance_

            dist = np.linalg.norm(empirical_mean - mu_posterior)**2 + \
                    np.trace(empirical_cov + cov_posterior - 2*sp.linalg.sqrtm( sp.linalg.sqrtm(cov_posterior)@empirical_cov@sp.linalg.sqrtm(cov_posterior)  ) )
            
            dist = np.sqrt(dist)
            wasserstein_dists[-1].append(dist)


            ########################################################################################################################
            # Check agreement between agents
            # empirical_mean_1 = np.mean(xs_per_agent[0], axis=0)
            # empirical_mean_2 = np.mean(xs_per_agent[1], axis=0)
            # empirical_cov_1  = EmpiricalCovariance(assume_centered=False).fit(xs_per_agent[0]).covariance_
            # empirical_cov_2  = EmpiricalCovariance(assume_centered=False).fit(xs_per_agent[1]).covariance_

            # dist = np.linalg.norm(empirical_mean_1 - empirical_mean_2)**2 + \
            #         np.trace(empirical_cov_1 + empirical_cov_2 - 2*sp.linalg.cholesky( sp.linalg.cholesky(empirical_cov_1)@empirical_cov_2@sp.linalg.cholesky(empirical_cov_1)  ) )
            
            # dist = np.sqrt(dist)
            # wasserstein_dists[0].append(dist)

            ########################################################################################################################

            # # Plot samples at end of deployment
            # plt.figure()                        
            # for aa in range(num_agents):
            #     colors = ['b', 'g', 'r', 'm', 'k']
            #     plt.plot(xs_per_agent[aa][:, 0], xs_per_agent[aa][:, 1], '*', color = colors[aa%len(colors)])
            # plt.savefig('./Results/Bay_Lin_Reg/DeSGLD/desgld_'+str(iter))      

            ########################################################################################################################
            # Main loop
            for trial in range(num_trials):

                x_per_agent = [xs_per_agent[aa][trial, :] for aa in range(num_agents)]
                v_per_agent = [vs_per_agent[aa][trial, :] for aa in range(num_agents)]

                grads = [grad_fi(x_per_agent[aa], X_per_agent[aa], y_per_agent[aa]) for aa in range(num_agents)]

                for aa in range(num_agents):
                    v_new = v_per_agent[aa] - eta*(gamma*v_per_agent[aa] + grads[aa] ) + np.sqrt(gamma*eta*2) * np.random.multivariate_normal(np.zeros(dim,), np.eye(dim))
                    
                    averaging_term = np.array([W[aa, aaa]*x_per_agent[aaa] for aaa in range(num_agents)])
                    xs_per_agent[aa][trial, :] = np.sum(averaging_term, axis=0) + eta*v_new

                    vs_per_agent[aa][trial, :] = v_new
 

    elif method == "D-ADMMS":

        ps_per_agent = [np.zeros((num_trials,dim)) for aa in range(num_agents)]
        
        rho = 5.

        # Define local problems 
        x_i = cp.Variable(dim)
        iterate_parameters = [cp.Parameter(dim) for aa in range(num_agents)]
        dual_parameters = [cp.Parameter(dim) for aa in range(num_agents)] 
        noise_parameters = [cp.Parameter(dim) for aa in range(num_agents)] 
        problems = []

        for aa in range(num_agents):
            # Define local convex problem
            objective = (1/(2*ksi**2)) * cp.norm(y_per_agent[aa] - x_i.T@X_per_agent[aa])**2 + (1/(2*lamda*num_agents)) * cp.norm(x_i)**2 \
                        + dual_parameters[aa].T@x_i \
                            + rho*cp.sum([cp.norm( x_i - 0.5*(iterate_parameters[aa]+iterate_parameters[j]) + noise_parameters[aa])**2 for j in range(len(list(graph.adj[aa])))])
            prob = cp.Problem(cp.Minimize(objective))
            problems.append(prob)

        # Main loop
        for iter in tqdm(range(num_iterations)):
            ########################################################################################################################
            # Compute Wasserstein distance of agents' iterates with target
            for aa in range(num_agents):
                empirical_mean = np.mean(xs_per_agent[aa], axis=0)
                empirical_cov  = EmpiricalCovariance(assume_centered=False).fit(xs_per_agent[aa]).covariance_

                dist = np.linalg.norm(empirical_mean - mu_posterior)**2 + \
                        np.trace(empirical_cov + cov_posterior - 2*sp.linalg.sqrtm( sp.linalg.sqrtm(cov_posterior)@empirical_cov@sp.linalg.sqrtm(cov_posterior)  ) )
                
                dist = np.sqrt(dist)
                wasserstein_dists[aa].append(dist)

            # Compute Wasserstein distance of average iterate
            average_iterates = []
            for trial in range(num_trials):
                trial_iterates = [xs_per_agent[aa][trial, :] for aa in range(num_agents)]
                average_iterate = np.mean(trial_iterates, axis=0)
                average_iterates.append(average_iterate)

            empirical_mean = np.mean(average_iterates, axis=0)
            empirical_cov  = EmpiricalCovariance(assume_centered=False).fit(average_iterates).covariance_

            dist = np.linalg.norm(empirical_mean - mu_posterior)**2 + \
                    np.trace(empirical_cov + cov_posterior - 2*sp.linalg.sqrtm( sp.linalg.sqrtm(cov_posterior)@empirical_cov@sp.linalg.sqrtm(cov_posterior)  ) )
            
            dist = np.sqrt(dist)
            wasserstein_dists[-1].append(dist)


            ########################################################################################################################
            # Check agreement between agents
            # if iter >= 1:
                # empirical_mean_1 = np.mean(xs_per_agent[0], axis=0)
                # empirical_mean_2 = np.mean(xs_per_agent[1], axis=0)
                # empirical_cov_1  = EmpiricalCovariance(assume_centered=False).fit(xs_per_agent[0]).covariance_
                # empirical_cov_2  = EmpiricalCovariance(assume_centered=False).fit(xs_per_agent[1]).covariance_

                # dist = np.linalg.norm(empirical_mean_1 - empirical_mean_2)**2 + \
                #         np.trace(empirical_cov_1 + empirical_cov_2 - 2*sp.linalg.cholesky( sp.linalg.cholesky(empirical_cov_1)@empirical_cov_2@sp.linalg.cholesky(empirical_cov_1)  ) )
                
                # dist = np.sqrt(dist)
                # wasserstein_dists[0].append(dist)

            ########################################################################################################################

            # # Plot samples at end of deployment
            # plt.figure()                        
            # for aa in range(num_agents):
            #     colors = ['b', 'g', 'r', 'm', 'k']
            #     plt.plot(xs_per_agent[aa][:, 0], xs_per_agent[aa][:, 1], '*', color = colors[aa%len(colors)])
            # plt.savefig('./Results/Bay_Lin_Reg/CADMM/cadmm'+str(iter))    

            ########################################################################################################################
            # Main loop
            for trial in range(num_trials):

                for aa in range(num_agents):
                    iterate_parameters[aa].value = xs_per_agent[aa][trial, :]
                    dual_parameters[aa].value = ps_per_agent[aa][trial, :]
                    if all_params_and_data["ADMM"] == False:
                        noise_parameters[aa].value = np.sqrt(2)/(2*rho)*random.normal(0., 1., (dim,))
                    else:
                        noise_parameters[aa].value = np.zeros(dim)

                for aa in range(num_agents):
                    problems[aa].solve()

                    if problems[aa].status != "optimal" and problems[aa].status != "optimal_inaccurate":
                        print(problems[aa].status)

                    xs_per_agent[aa][trial, :] = x_i.value 

                # Update dual variables
                for aa in range(num_agents):
                    term = np.array([xs_per_agent[aa][trial, :]-xs_per_agent[j][trial, :] for j in list(graph.adj[aa])])
                    ps_per_agent[aa][trial, :] += rho*np.sum(term, axis=0)
                                       

    return wasserstein_dists



    
#######################################################
###########          Results                ###########
agent_samples = 1


# Full graph
plt.subplot(1, 3, 1)

results = {}

all_params_and_data['graph'] = nx.complete_graph(num_agents)
wasserstein_dists = deploy_dynamics("D-SGLD", all_params_and_data)
results['D-SGLD'] = wasserstein_dists
for aa in range(agent_samples):
    plt.plot(range(len(wasserstein_dists[aa])), wasserstein_dists[aa], label="D-SGLD, ag")
plt.plot(range(len(wasserstein_dists[-1])), wasserstein_dists[-1], label="D-SGLD, avg")

all_params_and_data['graph'] = nx.complete_graph(num_agents)
all_params_and_data["delta_1"] = 0.55
all_params_and_data["delta_2"] = 0.05
wasserstein_dists = deploy_dynamics("D-ULA", all_params_and_data)
results['D-ULA'] = wasserstein_dists
for aa in range(agent_samples):
    plt.plot(range(len(wasserstein_dists[aa])), wasserstein_dists[aa], label="D-ULA, ag")
plt.plot(range(len(wasserstein_dists[-1])), wasserstein_dists[-1], label="D-ULA, avg")

all_params_and_data['graph'] = nx.complete_graph(num_agents)
wasserstein_dists = deploy_dynamics("D-SGHMC", all_params_and_data)
results['D-SGHMC'] = wasserstein_dists
for aa in range(agent_samples):
    plt.plot(range(len(wasserstein_dists[aa])), wasserstein_dists[aa], label="D-SGHMC, ag")
plt.plot(range(len(wasserstein_dists[-1])), wasserstein_dists[-1], label="D-SGHMC, avg")

all_params_and_data['graph'] = nx.complete_graph(num_agents)
all_params_and_data["ADMM"] = False
wasserstein_dists = deploy_dynamics("D-ADMMS", all_params_and_data)
results['D-ADMMS'] = wasserstein_dists
for aa in range(agent_samples):
    plt.plot(range(len(wasserstein_dists[aa])), wasserstein_dists[aa], label="D-ADMMS, ag")
plt.plot(range(len(wasserstein_dists[-1])), wasserstein_dists[-1], label="D-ADMMS, avg")

all_params_and_data['graph'] = nx.complete_graph(num_agents)
all_params_and_data["ADMM"] = True
wasserstein_dists = deploy_dynamics("D-ADMMS", all_params_and_data)
results['ADMM'] = wasserstein_dists
for aa in range(agent_samples):
    plt.plot(range(len(wasserstein_dists[aa])), wasserstein_dists[aa], linestyle='dashed', label="ADMM, ag")
plt.plot(range(len(wasserstein_dists[-1])), wasserstein_dists[-1], linestyle='dashed', label="ADMM, avg")

plt.xlabel("Iteration")
plt.ylabel('Wasserstein Distance')
plt.title("Fully connected")
plt.legend()

with open('Results/bay_lin_reg - full_graph - num_agents = '+str(num_agents)+'.pkl', 'wb') as fp:
    pickle.dump(results, fp)


# Cyclic graph
plt.subplot(1,2,1)

results = {}

all_params_and_data['graph'] = nx.cycle_graph(num_agents)
wasserstein_dists = deploy_dynamics("D-SGLD", all_params_and_data)
results['D-SGLD'] = wasserstein_dists
for aa in range(agent_samples):
    plt.plot(range(len(wasserstein_dists[aa])), wasserstein_dists[aa], label="D-SGLD, ag")
plt.plot(range(len(wasserstein_dists[-1])), wasserstein_dists[-1], label="D-SGLD, avg")

all_params_and_data['graph'] = nx.cycle_graph(num_agents)
all_params_and_data["delta_1"] = 0.05
all_params_and_data["delta_2"] = 0.05
wasserstein_dists = deploy_dynamics("D-ULA", all_params_and_data)
results['D-ULA'] = wasserstein_dists
for aa in range(agent_samples):
    plt.plot(range(len(wasserstein_dists[aa])), wasserstein_dists[aa], label="D-ULA, ag")
plt.plot(range(len(wasserstein_dists[-1])), wasserstein_dists[-1], label="D-ULA, avg")

all_params_and_data['graph'] = nx.cycle_graph(num_agents)
wasserstein_dists = deploy_dynamics("D-SGHMC", all_params_and_data)
results['D-SGHMC'] = wasserstein_dists
for aa in range(agent_samples):
    plt.plot(range(len(wasserstein_dists[aa])), wasserstein_dists[aa], label="D-SGHMC, ag")
plt.plot(range(len(wasserstein_dists[-1])), wasserstein_dists[-1], label="D-SGHMC, avg")

all_params_and_data['graph'] = nx.cycle_graph(num_agents)
all_params_and_data["ADMM"] = False
wasserstein_dists = deploy_dynamics("D-ADMMS", all_params_and_data)
results['D-ADMMS'] = wasserstein_dists
for aa in range(agent_samples):
    plt.plot(range(len(wasserstein_dists[aa])), wasserstein_dists[aa], label="D-ADMMS, ag")
plt.plot(range(len(wasserstein_dists[-1])), wasserstein_dists[-1], label="D-ADMMS, avg")

all_params_and_data['graph'] = nx.cycle_graph(num_agents)
all_params_and_data["ADMM"] = True
wasserstein_dists = deploy_dynamics("D-ADMMS", all_params_and_data)
results['ADMM'] = wasserstein_dists
for aa in range(agent_samples):
    plt.plot(range(len(wasserstein_dists[aa])), wasserstein_dists[aa], linestyle='dashed', label="ADMM, ag")
plt.plot(range(len(wasserstein_dists[-1])), wasserstein_dists[-1], linestyle='dashed', label="ADMM, avg")

plt.xlabel("Iteration")
plt.ylabel('Wasserstein Distance')
plt.title("Cyclic")
plt.legend()

with open('Results/bay_lin_reg - cyclic_graph - num_agents = '+str(num_agents)+'.pkl', 'wb') as fp:
    pickle.dump(results, fp)


# Graph with no edges
plt.subplot(1,2,2)

results = {}

G = nx.Graph()
G.add_nodes_from([_ for _ in range(num_agents)])
all_params_and_data['graph'] = G
wasserstein_dists = deploy_dynamics("D-SGLD", all_params_and_data)
results['D-SGLD'] = wasserstein_dists
for aa in range(agent_samples):
    plt.plot(range(len(wasserstein_dists[aa])), wasserstein_dists[aa], label="D-SGLD, ag")
plt.plot(range(len(wasserstein_dists[-1])), wasserstein_dists[-1], label="D-SGLD, avg")

G = nx.Graph()
G.add_nodes_from([_ for _ in range(num_agents)])
all_params_and_data['graph'] = G
all_params_and_data["delta_1"] = 0.05
all_params_and_data["delta_2"] = 0.05
wasserstein_dists = deploy_dynamics("D-ULA", all_params_and_data)
results['D-ULA'] = wasserstein_dists
for aa in range(agent_samples):
    plt.plot(range(len(wasserstein_dists[aa])), wasserstein_dists[aa], label="D-ULA, ag")
plt.plot(range(len(wasserstein_dists[-1])), wasserstein_dists[-1], label="D-ULA, avg")

G = nx.Graph()
G.add_nodes_from([_ for _ in range(num_agents)])
all_params_and_data['graph'] = G
wasserstein_dists = deploy_dynamics("D-SGHMC", all_params_and_data)
results['D-SGHMC'] = wasserstein_dists
for aa in range(agent_samples):
    plt.plot(range(len(wasserstein_dists[aa])), wasserstein_dists[aa], label="D-SGHMC, ag")
plt.plot(range(len(wasserstein_dists[-1])), wasserstein_dists[-1], label="D-SGHMC, avg")

G = nx.Graph()
G.add_nodes_from([_ for _ in range(num_agents)])
all_params_and_data['graph'] = G
all_params_and_data["ADMM"] = False
wasserstein_dists = deploy_dynamics("D-ADMMS", all_params_and_data)
results['D-ADMMS'] = wasserstein_dists
for aa in range(agent_samples):
    plt.plot(range(len(wasserstein_dists[aa])), wasserstein_dists[aa], label="D-ADMMS, ag")
plt.plot(range(len(wasserstein_dists[-1])), wasserstein_dists[-1], label="D-ADMMS, avg")

G = nx.Graph()
G.add_nodes_from([_ for _ in range(num_agents)])
all_params_and_data['graph'] = G
all_params_and_data["ADMM"] = True
wasserstein_dists = deploy_dynamics("D-ADMMS", all_params_and_data)
results['ADMM'] = wasserstein_dists
for aa in range(agent_samples):
    plt.plot(range(len(wasserstein_dists[aa])), wasserstein_dists[aa], linestyle='dashed', label="ADMM, ag")
plt.plot(range(len(wasserstein_dists[-1])), wasserstein_dists[-1], linestyle='dashed', label="ADMM, avg")

plt.xlabel("Iteration")
plt.ylabel('Wasserstein Distance')
plt.title("No edges")
plt.legend()

with open('Results/bay_lin_reg - no_edges - num_agents = '+str(num_agents)+'.pkl', 'wb') as fp:
    pickle.dump(results, fp)

plt.suptitle('Results for '+str(num_agents)+' agents')
plt.show()
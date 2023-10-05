"""
    In this file, we regenerate the results of
        "Decentralized Stochastic Gradient Langevin Dynamics and Hamiltonian Monte Carlo"
    in the case of Bayesian logistic regression. 

    We implement D-ULA from
        "A Decentralized Approach to Bayesian Learning".

    Further, we implement our ADMM-based sampling algorithm.
"""

#######################################################
# Add dependencies
import numpy as np
import cvxpy as cp
import networkx as nx
import random as rnd
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
dim = 3

# Define network connectivity
num_agents = 50
graph = nx.complete_graph(num_agents)

# Define number of samples per agent
num_samples_per_agent = 50

# Data generation
lamda = 10

X_per_agent = []
y_per_agent = []

x = random.multivariate_normal(np.zeros(dim), lamda*np.eye(dim) )

for aa in range(num_agents):
    X_agent = random.multivariate_normal(np.zeros(dim), 20*np.eye(dim), size=(num_samples_per_agent,)).T

    X_per_agent.append(X_agent)

    ys = []
    for sample in range(num_samples_per_agent): 
        p = np.random.uniform(low=0., high=1.)
        if p<= 1/(1+np.exp(-x.T@X_agent[:, sample])):
            ys.append(1)
        else:
            ys.append(0)

    y_per_agent.append(ys)

X = np.hstack(X_per_agent)
y = np.hstack(y_per_agent)

# print(X_per_agent[0].shape) # (2,num_samples_per_agent)
# print(y_per_agent[0].shape) # (1,num_samples_per_agent)

# Define gradients
grad_fi = lambda _x, _X, _y: np.array([np.sum([ _X[0, _]*np.exp(_x.T@_X[:, _])/(1+np.exp(_x.T@_X[:, _]))    for _ in range(_X.shape[1])]),\
                                    np.sum([ _X[1, _]*np.exp(_x.T@_X[:, _])/(1+np.exp(_x.T@_X[:, _]))    for _ in range(_X.shape[1])]),\
                                    np.sum([ _X[2, _]*np.exp(_x.T@_X[:, _])/(1+np.exp(_x.T@_X[:, _]))    for _ in range(_X.shape[1])])    ])\
                        + np.array([np.sum([-_X[0, _] for _ in range(_X.shape[1]) if _y[_]==1]),\
                                    np.sum([-_X[1, _] for _ in range(_X.shape[1]) if _y[_]==1]),\
                                    np.sum([-_X[2, _] for _ in range(_X.shape[1]) if _y[_]==1])]) \
                        + x/(lamda*num_agents)

# Define step size 
eta_dsgld = 0.0003
eta_dsghmc = 0.02

gamma = 30.

alpha_0 = 0.00082
beta_0  = 0.48
b_1 = 230
b_2 = 230
delta_1 = 0.05
delta_2 =  0.05

# Define number of iterations
num_iterations = 20

# Define number of trials
num_trials = 100

# Include all parameters in dictionary
all_params_and_data = {}
all_params_and_data['graph'] = graph
all_params_and_data['dim'] = dim
all_params_and_data['num_agents'] = num_agents
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
    Xs = np.hstack(X_per_agent)
    ys = np.hstack(y_per_agent)

    # Define doubly stochastic matrix based on connectivity
    W = np.zeros((num_agents, num_agents))

    for edge in list(graph.edges):
        W[edge[0], edge[1]] = 1/(np.maximum(graph.degree[edge[0]], graph.degree[edge[1]])+1)
        W[edge[1], edge[0]] = W[edge[0], edge[1]]
    for aa in range(num_agents):
        W[aa, aa] = 1 - np.sum(W[aa, :])

    W_ext = np.kron(W, np.eye(dim))

    # Define output variable
    accuracies = [[[] for _ in range(num_iterations)] for aa in range(num_agents)]

    # Initialize trial samples
    xs_per_agent = [np.random.multivariate_normal(np.zeros((dim,)), np.eye(dim), size=(num_trials,)) for aa in range(num_agents)]


    # Choose sampling method
    if method == "D-SGLD":

        eta = eta_dsgld

        # Main loop
        for iter in tqdm(range(num_iterations)):
            ########################################################################################################################
            # Compute accuracies with respect to dataset
            for trial in range(num_trials):
                for aa in range(num_agents):
                    x_k = xs_per_agent[aa][trial, :]

                    pred_y = 1/(1+np.exp(-x_k.T@Xs))
                    pred_y[pred_y >= 0.5] = 1
                    pred_y[pred_y <  0.5] = 0
 
                    accuracies[aa][iter].append(np.mean(pred_y==ys))

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
            # Compute accuracies with respect to dataset
            for trial in range(num_trials):
                for aa in range(num_agents):
                    x_k = xs_per_agent[aa][trial, :]

                    pred_y = 1/(1+np.exp(-x_k.T@Xs))
                    pred_y[pred_y >= 0.5] = 1
                    pred_y[pred_y <  0.5] = 0
 
                    accuracies[aa][iter].append(np.mean(pred_y==ys))

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
            # Compute accuracies with respect to dataset
            for trial in range(num_trials):
                for aa in range(num_agents):
                    x_k = xs_per_agent[aa][trial, :]

                    pred_y = 1/(1+np.exp(-x_k.T@Xs))
                    pred_y[pred_y >= 0.5] = 1
                    pred_y[pred_y <  0.5] = 0
 
                    accuracies[aa][iter].append(np.mean(pred_y==ys))

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
            objective = cp.sum([cp.logistic(-x_i.T@X_per_agent[aa][:, _]) for _ in range(num_samples_per_agent) if y_per_agent[aa][_]==1]) \
                        + cp.sum([cp.logistic(x_i.T@X_per_agent[aa][:, _]) for _ in range(num_samples_per_agent) if y_per_agent[aa][_]==0]) \
                        + (1/(2*lamda*num_agents)) * cp.norm(x_i)**2 \
                        + dual_parameters[aa].T@x_i \
                        + rho*cp.sum([cp.norm( x_i - 0.5*(iterate_parameters[aa]+iterate_parameters[j]) + noise_parameters[aa])**2 for j in range(len(list(graph.adj[aa])))])
            prob = cp.Problem(cp.Minimize(1e-7*objective))
            problems.append(prob)

        # Main loop
        for iter in tqdm(range(num_iterations)):
            ########################################################################################################################
            # Compute accuracies with respect to dataset
            for trial in range(num_trials):
                for aa in range(num_agents):
                    x_k = xs_per_agent[aa][trial, :]

                    pred_y = 1/(1+np.exp(-x_k.T@Xs))
                    pred_y[pred_y >= 0.5] = 1
                    pred_y[pred_y <  0.5] = 0
 
                    accuracies[aa][iter].append(np.mean(pred_y==ys))

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
                    term = np.array([(xs_per_agent[aa][trial, :]-xs_per_agent[j][trial, :]) for j in list(graph.adj[aa])])
                    ps_per_agent[aa][trial, :] += rho*np.sum(term, axis=0)
                                                
                                                

    return accuracies



    
#######################################################
###########          Results                ###########
agent_samples = num_agents

# Helper function
def stds(accuracies, aa):
    res = []
    for _ in range(num_iterations):
        res.append(np.std(accuracies[aa][_]))

    return res


# Full graph
plt.subplot(1, 3, 1)

results = {}

all_params_and_data['graph'] = nx.complete_graph(num_agents)
accuracies = deploy_dynamics("D-SGLD", all_params_and_data)
results['D-SGLD'] = accuracies
for aa in range(agent_samples):
    means = [np.mean(accuracies[aa][_]) for _ in range(len(accuracies[aa]))]
    if aa == 0:
        plt.plot(range(len(accuracies[aa])), means, label="D-SGLD, ag", color="blue")
    else:
        plt.plot(range(len(accuracies[aa])), means, color="blue")
    plt.fill_between(range(len(accuracies[aa])), np.array(means)+np.array(stds(accuracies, aa)), np.array(means)-np.array(stds(accuracies, aa)), alpha=.3, color="blue")

all_params_and_data['graph'] = nx.complete_graph(num_agents)
all_params_and_data["delta_1"] = 0.9
all_params_and_data["delta_2"] = 0.9
accuracies = deploy_dynamics("D-ULA", all_params_and_data)
results['D-ULA'] = accuracies
for aa in range(agent_samples):
    means = [np.mean(accuracies[aa][_]) for _ in range(len(accuracies[aa]))]
    if aa == 0:
        plt.plot(range(len(accuracies[aa])), means, label="D-ULA, ag", color="magenta")
    else:
        plt.plot(range(len(accuracies[aa])), means, color="magenta")
    plt.fill_between(range(len(accuracies[aa])), np.array(means)+np.array(stds(accuracies, aa)), np.array(means)-np.array(stds(accuracies, aa)), alpha=.3, color="magenta")

all_params_and_data['graph'] = nx.complete_graph(num_agents)
accuracies = deploy_dynamics("D-SGHMC", all_params_and_data)
results['D-SGHMC'] = accuracies
for aa in range(agent_samples):
    means = [np.mean(accuracies[aa][_]) for _ in range(len(accuracies[aa]))]
    if aa == 0:
        plt.plot(range(len(accuracies[aa])), means, label="D-SGHMC, ag", color="green")
    else:
        plt.plot(range(len(accuracies[aa])), means, color="green")
    plt.fill_between(range(len(accuracies[aa])), np.array(means)+np.array(stds(accuracies, aa)), np.array(means)-np.array(stds(accuracies, aa)), alpha=.3, color="green")

all_params_and_data['graph'] = nx.complete_graph(num_agents)
all_params_and_data['ADMM'] = False
accuracies = deploy_dynamics("D-ADMMS", all_params_and_data)
results['D-ADMMS'] = accuracies
for aa in range(agent_samples):
    means = [np.mean(accuracies[aa][_]) for _ in range(len(accuracies[aa]))]
    if aa == 0:
        plt.plot(range(len(accuracies[aa])), means, label="D-ADMMS, ag", color="orange")
    else:
        plt.plot(range(len(accuracies[aa])), means, color="orange")
    plt.fill_between(range(len(accuracies[aa])), np.array(means)+np.array(stds(accuracies, aa)), np.array(means)-np.array(stds(accuracies, aa)), alpha=.3, color="orange")

all_params_and_data['graph'] = nx.complete_graph(num_agents)
all_params_and_data['ADMM'] = True
accuracies = deploy_dynamics("D-ADMMS", all_params_and_data)
results['ADMM'] = accuracies
for aa in range(agent_samples):
    means = [np.mean(accuracies[aa][_]) for _ in range(len(accuracies[aa]))]
    if aa == 0:
        plt.plot(range(len(accuracies[aa])), means, label="ADMM, ag", color="cyan")
    else:
        plt.plot(range(len(accuracies[aa])), means, color="cyan")
    plt.fill_between(range(len(accuracies[aa])), np.array(means)+np.array(stds(accuracies, aa)), np.array(means)-np.array(stds(accuracies, aa)), alpha=.3, color="cyan")


with open('Results/bay_log_reg - full_graph - num_agents = '+str(num_agents)+'.pkl', 'wb') as fp:
    pickle.dump(results, fp)

plt.xlabel("Iteration")
plt.ylabel('Accuracy')
plt.title("Fully connected")
plt.legend()


# Cyclic graph
plt.subplot(1,3,2)

results = {}

all_params_and_data['graph'] = nx.cycle_graph(num_agents)
accuracies = deploy_dynamics("D-SGLD", all_params_and_data)
results['D-SGLD'] = accuracies
for aa in range(agent_samples):
    means = [np.mean(accuracies[aa][_]) for _ in range(len(accuracies[aa]))]
    if aa == 0:
        plt.plot(range(len(accuracies[aa])), means, label="D-SGLD, ag", color="blue")
    else:
        plt.plot(range(len(accuracies[aa])), means, color="blue")
    plt.fill_between(range(len(accuracies[aa])), np.array(means)+np.array(stds(accuracies, aa)), np.array(means)-np.array(stds(accuracies, aa)), alpha=.3, color="blue")

all_params_and_data['graph'] = nx.cycle_graph(num_agents)
all_params_and_data["delta_1"] = 0.05
all_params_and_data["delta_2"] = 0.05
accuracies = deploy_dynamics("D-ULA", all_params_and_data)
results['D-ULA'] = accuracies
for aa in range(agent_samples):
    means = [np.mean(accuracies[aa][_]) for _ in range(len(accuracies[aa]))]
    if aa == 0:
        plt.plot(range(len(accuracies[aa])), means, label="D-ULA, ag", color="magenta")
    else:
        plt.plot(range(len(accuracies[aa])), means, color="magenta")
    plt.fill_between(range(len(accuracies[aa])), np.array(means)+np.array(stds(accuracies, aa)), np.array(means)-np.array(stds(accuracies, aa)), alpha=.3, color="magenta")

all_params_and_data['graph'] = nx.cycle_graph(num_agents)
accuracies = deploy_dynamics("D-SGHMC", all_params_and_data)
results['D-SGHMC'] = accuracies
for aa in range(agent_samples):
    means = [np.mean(accuracies[aa][_]) for _ in range(len(accuracies[aa]))]
    if aa == 0:
        plt.plot(range(len(accuracies[aa])), means, label="D-SGHMC, ag", color="green")
    else:
        plt.plot(range(len(accuracies[aa])), means, color="green")
    plt.fill_between(range(len(accuracies[aa])), np.array(means)+np.array(stds(accuracies, aa)), np.array(means)-np.array(stds(accuracies, aa)), alpha=.3, color="green")

all_params_and_data['graph'] = nx.cycle_graph(num_agents)
all_params_and_data["ADMM"] = False
accuracies = deploy_dynamics("D-ADMMS", all_params_and_data)
results['D-ADMMS'] = accuracies
for aa in range(agent_samples):
    means = [np.mean(accuracies[aa][_]) for _ in range(len(accuracies[aa]))]
    if aa == 0:
        plt.plot(range(len(accuracies[aa])), means, label="D-ADMMS, ag", color="orange")
    else:
        plt.plot(range(len(accuracies[aa])), means, color="orange")
    plt.fill_between(range(len(accuracies[aa])), np.array(means)+np.array(stds(accuracies, aa)), np.array(means)-np.array(stds(accuracies, aa)), alpha=.3, color="orange")

all_params_and_data['graph'] = nx.cycle_graph(num_agents)
all_params_and_data["ADMM"] = True
accuracies = deploy_dynamics("D-ADMMS", all_params_and_data)
results['ADMM'] = accuracies
for aa in range(agent_samples):
    means = [np.mean(accuracies[aa][_]) for _ in range(len(accuracies[aa]))]
    if aa == 0:
        plt.plot(range(len(accuracies[aa])), means, label="ADMM, ag", color="cyan")
    else:
        plt.plot(range(len(accuracies[aa])), means, color="cyan")
    plt.fill_between(range(len(accuracies[aa])), np.array(means)+np.array(stds(accuracies, aa)), np.array(means)-np.array(stds(accuracies, aa)), alpha=.3, color="cyan")


with open('Results/bay_log_reg - cyclic_graph - num_agents = '+str(num_agents)+'.pkl', 'wb') as fp:
    pickle.dump(results, fp)

plt.xlabel("Iteration")
plt.ylabel('Accuracy')
plt.title("Cyclic")
plt.legend()


# Graph with no edges
plt.subplot(1,3,3)

results = {}

G = nx.Graph()
G.add_nodes_from([_ for _ in range(num_agents)])
all_params_and_data['graph'] = G
accuracies = deploy_dynamics("D-SGLD", all_params_and_data)
results['D-SGLD'] = accuracies
for aa in range(agent_samples):
    means = [np.mean(accuracies[aa][_]) for _ in range(len(accuracies[aa]))]
    if aa == 0:
        plt.plot(range(len(accuracies[aa])), means, label="D-SGLD, ag", color="blue")
    else:
        plt.plot(range(len(accuracies[aa])), means, color="blue")
    plt.fill_between(range(len(accuracies[aa])), np.array(means)+np.array(stds(accuracies, aa)), np.array(means)-np.array(stds(accuracies, aa)), alpha=.3, color="blue")

G = nx.Graph()
G.add_nodes_from([_ for _ in range(num_agents)])
all_params_and_data['graph'] = G
all_params_and_data["delta_1"] = 0.05
all_params_and_data["delta_2"] = 0.05
accuracies = deploy_dynamics("D-ULA", all_params_and_data)
results['D-ULA'] = accuracies
for aa in range(agent_samples):
    means = [np.mean(accuracies[aa][_]) for _ in range(len(accuracies[aa]))]
    if aa == 0:
        plt.plot(range(len(accuracies[aa])), means, label="D-ULA, ag", color="magenta")
    else:
        plt.plot(range(len(accuracies[aa])), means, color="magenta")

    plt.fill_between(range(len(accuracies[aa])), np.array(means)+np.array(stds(accuracies, aa)), np.array(means)-np.array(stds(accuracies, aa)), alpha=.3, color="magenta")

G = nx.Graph()
G.add_nodes_from([_ for _ in range(num_agents)])
all_params_and_data['graph'] = G
accuracies = deploy_dynamics("D-SGHMC", all_params_and_data)
results['D-SGHMC'] = accuracies
for aa in range(agent_samples):
    means = [np.mean(accuracies[aa][_]) for _ in range(len(accuracies[aa]))]
    if aa == 0:
        plt.plot(range(len(accuracies[aa])), means, label="D-SGHMC, ag", color="green")
    else:
        plt.plot(range(len(accuracies[aa])), means, color="green")
    plt.fill_between(range(len(accuracies[aa])), np.array(means)+np.array(stds(accuracies, aa)), np.array(means)-np.array(stds(accuracies, aa)), alpha=.3, color="green")

G = nx.Graph()
G.add_nodes_from([_ for _ in range(num_agents)])
all_params_and_data['graph'] = G
all_params_and_data["ADMM"] = False
accuracies = deploy_dynamics("D-ADMMS", all_params_and_data)
results['D-ADMMS'] = accuracies
for aa in range(agent_samples):
    means = [np.mean(accuracies[aa][_]) for _ in range(len(accuracies[aa]))]
    if aa == 0:
        plt.plot(range(len(accuracies[aa])), means, label="D-ADMMS, ag", color="orange")
    else:
        plt.plot(range(len(accuracies[aa])), means, color="orange")
    plt.fill_between(range(len(accuracies[aa])), np.array(means)+np.array(stds(accuracies, aa)), np.array(means)-np.array(stds(accuracies, aa)), alpha=.3, color="orange")

G = nx.Graph()
G.add_nodes_from([_ for _ in range(num_agents)])
all_params_and_data['graph'] = G
all_params_and_data["ADMM"] = True
accuracies = deploy_dynamics("D-ADMMS", all_params_and_data)
results['ADMM'] = accuracies
for aa in range(agent_samples):
    means = [np.mean(accuracies[aa][_]) for _ in range(len(accuracies[aa]))]
    if aa == 0:
        plt.plot(range(len(accuracies[aa])), means, label="ADMM, ag", color="cyan")
    else:
        plt.plot(range(len(accuracies[aa])), means, color="cyan")
    plt.fill_between(range(len(accuracies[aa])), np.array(means)+np.array(stds(accuracies, aa)), np.array(means)-np.array(stds(accuracies, aa)), alpha=.3, color="cyan")

with open('Results/bay_log_reg - no_edges - num_agents = '+str(num_agents)+'.pkl', 'wb') as fp:
    pickle.dump(results, fp)

plt.xlabel("Iteration")
plt.ylabel('Accuracy')
plt.title("No edges")
plt.legend(loc='lower right')

plt.suptitle('Results for '+str(num_agents)+' agents')
plt.show()
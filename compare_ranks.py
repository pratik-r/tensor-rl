from functions import *
from tqdm import tqdm
import numpy as np
import tensorly as tl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run(seed,alpha_dirs,Xi,nactions,d1,d2,d3,thetahat_qlearn,R1,reg_rank_seq,n_episodes,episode_length,n_samples,gamma,eps_policy,noise):
    np.random.seed(seed)
    px = (nactions,d1,d2,d3)
    bpolicy = np.ones(nactions) / nactions # uniformly random behavior policy
    
    Data = gen_data(n_episodes,episode_length,bpolicy,alpha_dirs,Xi,nactions,d1,d2,d3,noise)
    dat = flatten(Data)
    states = [d[0] for d in dat]
    actions = [d[1] for d in dat]
    rewards = np.array([d[2] for d in dat])
    states_ = [d[3] for d in dat]
    features = [phi_fn(s,a,nactions) for s,a in zip(states,actions)]
    features_ = [sum([policy(s, thetahat_qlearn, nactions, eps_policy)[a]*phi_fn(s,a,nactions) for a in range(nactions)]) for s in states_]
    
    states_eval = [np.random.dirichlet(alpha_dirs[np.random.choice(nactions)]).reshape(d1,d2,d3) for _ in range(n_samples)]
    mc_values = value_mc(states_eval,gamma,1e-10,Xi,alpha_dirs,thetahat_qlearn,nactions,d1,d2,d3,eps_policy)
    out = dict()
    for reg_rank in reg_rank_seq:
        model_tIVreg = tensor_IVreg(features, features_, rewards, nactions, reg_rank, gamma, eps=1e-3)
        thetas_tIVreg = model_tIVreg['thetas']
        Theta_tIVreg = tl.cp_to_tensor([model_tIVreg['weights'],model_tIVreg['thetas']])
        est_values = np.array([sum([policy(state, thetahat_qlearn, nactions, eps_policy)[a]*tl.tenalg.inner(Theta_tIVreg, phi_fn(state, a, nactions)) for a in range(nactions)]) for state in states_eval])
        out[reg_rank] = np.linalg.norm(mc_values - est_values) / np.sqrt(n_samples)
    return out
    
if __name__ == "__main__":
    niter = 50
    nactions = 2
    d1 = d2 = d3 = 10
    R1_seq = [1,2,3]
    reg_rank_seq = 1+np.arange(5)
    n_episodes = 1000
    episode_length = 50
    n_obs_q = int(1e5)
    n_samples = 1000
    w_xi = 10
    gamma = 0.9
    alpha_q = 1e-2
    eps_policy = 0.2
    noise = 0.1

    # fix true parameters and learn optimal policy
    np.random.seed(2022)
    px = (nactions,d1,d2,d3)
    alpha_dirs = [np.random.random(d1*d2*d3) for _ in range(nactions)]

    out = dict()
    for R1 in R1_seq:
        xis_list = [[np.random.normal(size=d) for d in px] for _ in range(R1)]
        xis_list = [[xi / np.linalg.norm(xi) for xi in xis] for xis in xis_list]
        Xi = sum([w_xi*tl.tensor(reduce(np.multiply.outer, xis)) for xis in xis_list])
        thetahat_qlearn = qlearn(alpha_dirs,Xi,px,nactions,d1,d2,d3,n_obs_q,gamma,alpha_q)
        out[R1+nactions] = []
        for seed in tqdm(range(niter)):
            out[R1+nactions] += [run(seed,alpha_dirs,Xi,nactions,d1,d2,d3,thetahat_qlearn,R1,reg_rank_seq,n_episodes,episode_length,n_samples,gamma,eps_policy,noise)]
            
    df = pd.concat({k:pd.DataFrame(v).stack().reset_index(level=1).rename(columns={'level_1':'regression_rank',0:'value_estimation_error'}) for k,v in out.items()}).reset_index(level=0).rename(columns={'level_0':'true_rank'}).reset_index(drop=True)

    plt.rcParams['figure.figsize'] = (16,9)
    plt.rcParams['figure.facecolor'] = 'white'
    plot = sns.catplot(kind='box', x='regression_rank', y='value_estimation_error', col='true_rank', data=df, sharey=False)
    plot.savefig('compare_ranks.png',bbox_inches='tight')
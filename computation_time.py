from functions import *
from tqdm import tqdm
import numpy as np
import tensorly as tl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

def run(seed,alpha_dirs,Xi,nactions,d1,d2,d3,thetahat_qlearn,R1,n_episodes_seq,episode_length,n_samples,gamma,eps_policy,noise):
    np.random.seed(seed)
    px = (nactions,d1,d2,d3)
    bpolicy = np.ones(nactions) / nactions # uniformly random behavior policy
    
    Data = gen_data(np.max(n_episodes_seq),episode_length,bpolicy,alpha_dirs,Xi,nactions,d1,d2,d3,noise)
    dat = flatten(Data)
    states = [d[0] for d in dat]
    actions = [d[1] for d in dat]
    rewards = np.array([d[2] for d in dat])
    states_ = [d[3] for d in dat]
    features = [phi_fn(s,a,nactions) for s,a in zip(states,actions)]
    features_ = [sum([policy(s, thetahat_qlearn, nactions, eps_policy)[a]*phi_fn(s,a,nactions) for a in range(nactions)]) for s in states_]
    
    rank = R1+nactions
    out = dict()
    for n_episodes in n_episodes_seq:
        start = time.time()
        model_tIVreg = tensor_IVreg_iter(features[:n_episodes], features_[:n_episodes], rewards[:n_episodes], nactions, rank, gamma, 1)
        end = time.time()
        out[n_episodes] = end-start
    return out
    
if __name__ == "__main__":
    niter = 50
    nactions = 2
    d1 = d2 = d3 = 10
    R1 = 1
    n_episodes_seq = np.arange(100,1100,100)
    episode_length = 50
    n_obs_q = int(1e5)
    n_samples = 1000
    w_xi = 10
    gamma = 0.9
    alpha_q = 1e-2
    eps_policy = 0.2
    noise = 1

    # fix true parameters and learn optimal policy
    np.random.seed(2022)
    px = (nactions,d1,d2,d3)
    alpha_dirs = [np.random.random(d1*d2*d3) for _ in range(nactions)]
    xis_list = [[np.random.normal(size=d) for d in px] for _ in range(R1)]
    xis_list = [[xi / np.linalg.norm(xi) for xi in xis] for xis in xis_list]
    Xi = sum([w_xi*tl.tensor(reduce(np.multiply.outer, xis)) for xis in xis_list])
    thetahat_qlearn = qlearn(alpha_dirs,Xi,px,nactions,d1,d2,d3,n_obs_q,gamma,alpha_q)

    out = []
    for seed in tqdm(range(niter)):
        out += [run(seed,alpha_dirs,Xi,nactions,d1,d2,d3,thetahat_qlearn,R1,n_episodes_seq,episode_length,n_samples,gamma,eps_policy,noise)]
        
    df = pd.DataFrame(out).stack().reset_index(level=1).rename(columns={'level_1':'num_observations',0:'computation_time_seconds'}).reset_index(drop=True)
    df['num_observations'] = df['num_observations']*episode_length

    plt.rcParams['figure.figsize'] = (16,9)
    plt.rcParams['figure.facecolor'] = 'white'
    sns.set_context("paper", font_scale=2)   
    plot = sns.pointplot(x='num_observations',y='computation_time_seconds',data=df,capsize=0.1)
    plot.get_figure().savefig('computation_time.png',bbox_inches='tight')

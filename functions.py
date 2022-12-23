import numpy as np
import tensorly as tl
from functools import reduce
import pandas as pd
import math
from copy import deepcopy

phi_fn = lambda s,a, nactions: tl.tensor(np.multiply.outer(np.eye(nactions)[a,:], s))

def cov_dir(alpha_dir):
    alpha0 = alpha_dir.sum()
    alpha_tilde = alpha_dir / alpha0
    return (np.diag(alpha_tilde) - np.outer(alpha_tilde, alpha_tilde)) / (alpha0 + 1)

def mu_dir(alpha_dir):
    return alpha_dir / alpha_dir.sum()

def flatten(l):
    return [item for sublist in l for item in sublist]

def gen_data(n_episodes,episode_length,bpolicy,alpha_dirs,Xi,nactions,d1,d2,d3,noise=0):
    Data = []
    for _ in range(n_episodes):
        data = []
        state = np.random.dirichlet(alpha_dirs[np.random.choice(nactions, p=bpolicy)]).reshape(d1,d2,d3)
        for i in range(episode_length):
            action = np.random.choice(nactions, p=bpolicy)
            phi = phi_fn(state,action,nactions)
            reward = tl.tenalg.inner(Xi, phi) + np.random.normal(scale=noise)
            state_ = np.random.dirichlet(alpha_dirs[action]).reshape(d1,d2,d3)
            data.append((state,action,reward,state_))
            state = state_
        Data.append(data)
    return Data
    
def qlearn(alpha_dirs,Xi,px,nactions,d1,d2,d3,n_obs,gamma,alpha=1e-2):
    thetahat_qlearn = np.zeros(np.prod(px))
    epsilons = 1-np.geomspace(start = 0.01, stop = 0.99, num=n_obs)
    for i in range(n_obs):
        state = np.random.dirichlet(alpha_dirs[np.random.choice(nactions)]).reshape(d1,d2,d3)
        if np.random.random() < epsilons[i]:
            action = np.random.choice(nactions)
        else:
            action = np.argmax([np.dot(phi_fn(state,a,nactions).reshape(-1),thetahat_qlearn) for a in range(nactions)])
        phi = phi_fn(state,action,nactions)
        reward = tl.tenalg.inner(Xi, phi)
        state_ = np.random.dirichlet(alpha_dirs[action]).reshape(d1,d2,d3)
        q_ = reward + gamma*np.max([np.dot(phi_fn(state_,a,nactions).reshape(-1), thetahat_qlearn) for a in range(nactions)])
        thetahat_qlearn += alpha*(q_ - np.dot(phi.reshape(-1), thetahat_qlearn))*phi.reshape(-1)
        state = state_
    return thetahat_qlearn
    
def lstdq(features,features_,rewards,gamma):
    N = len(features)
    Phi = np.array([f.reshape(-1) for f in features])
    Phi_ = np.array([f.reshape(-1) for f in features_])
    Ahat = 1/N * Phi.T @ (Phi - gamma*Phi_)
    bhat = (Phi * rewards[:,np.newaxis]).mean(axis=0)
    return np.linalg.inv(Ahat)@bhat
    
def fqe(features, features_, rewards, gamma, eps=1e-3):
    N = len(features); px = features[0].shape
    Xarr = np.array([f.reshape(-1) for f in features])
    Xarr_ = np.array([f.reshape(-1) for f in features_])
    XTXinv = np.linalg.pinv(Xarr.T@Xarr)
    theta = np.random.normal(size=np.prod(px))
    y = rewards + gamma*Xarr_@theta
    loss_old = 1/np.sqrt(N) * np.linalg.norm(y - Xarr@theta)
    while True:
        y = rewards + gamma*Xarr_@theta
        theta = np.linalg.multi_dot([XTXinv,Xarr.T,y])
        loss_new = 1/np.sqrt(N) * np.linalg.norm(y - Xarr@theta)
        if abs(loss_old - loss_new) / np.min([1,loss_old]) < eps:
            break
        loss_old = loss_new
    return {'theta': theta, 'loss': loss_new}
    
def tensor_IVreg(features, features_, rewards, nactions, rank, gamma, eps):
    px = features[0].shape; D = len(px); N=len(features)
    thetas = [np.random.normal(size=(p,rank)) for p in px]
    thetas = [theta / np.linalg.norm(theta,axis=0) for theta in thetas]
    weights = np.ones(rank)
    Theta = tl.cp_to_tensor([weights, thetas])
    X = [f-gamma*f_ for f,f_ in zip(features, features_)]
    Fmat = np.array([f.reshape(-1) for f in features])
    y = Fmat.T @ rewards
    delta = rewards - np.array([tl.tenalg.inner(x, Theta) for x in X])
    loss_old = 1/np.sqrt(N) * np.sqrt(np.linalg.multi_dot([delta.T, Fmat, Fmat.T, delta]))

    while True:
        for d in range(D):
            Xreg = np.array([(tl.unfold(x,d) @ tl.tenalg.khatri_rao([thetas[d_] for d_ in range(D) if d_ != d])).reshape(-1) for x in X])
            Sigma = Fmat.T @ Xreg
            thetas[d] = tl.vec_to_tensor(np.linalg.multi_dot([np.linalg.pinv(Sigma.T @ Sigma), Sigma.T, y]), shape=thetas[d].shape)
        thetas = [theta / np.linalg.norm(theta,axis=0) for theta in thetas]
        Thetas_unscaled = [tl.cp_to_tensor([np.ones(1),[theta[:,[r]] for theta in thetas]]) for r in range(rank)]
        XregW = np.array([[tl.tenalg.inner(Theta, x) for Theta in Thetas_unscaled] for x in X])
        SigmaW = Fmat.T @ XregW
        weights = np.linalg.multi_dot([np.linalg.pinv(SigmaW.T @ SigmaW), SigmaW.T, y])
        Theta = tl.cp_to_tensor([weights, thetas])
        delta = rewards - np.array([tl.tenalg.inner(x, Theta) for x in X])
        loss_new = 1/np.sqrt(N) * np.sqrt(np.linalg.multi_dot([delta.T, Fmat, Fmat.T, delta]))
        loss_diff = np.abs(loss_old - loss_new) / np.min([loss_old, 1])
        if loss_diff < eps:
            break
        loss_old = loss_new
    
    return {'thetas':thetas, 'weights':weights, 'loss':loss_new}

def tensor_IVreg_iter(features, features_, rewards, nactions, rank, gamma, niter_reg):
    px = features[0].shape; D = len(px); N=len(features)
    thetas = [np.random.normal(size=(p,rank)) for p in px]
    thetas = [theta / np.linalg.norm(theta,axis=0) for theta in thetas]
    weights = np.ones(rank)
    Theta = tl.cp_to_tensor([weights, thetas])
    X = [f-gamma*f_ for f,f_ in zip(features, features_)]
    Fmat = np.array([f.reshape(-1) for f in features])
    y = Fmat.T @ rewards
    delta = rewards - np.array([tl.tenalg.inner(x, Theta) for x in X])
    loss = 1/np.sqrt(N) * np.sqrt(np.linalg.multi_dot([delta.T, Fmat, Fmat.T, delta]))
    loss_dict = {0:loss}
    thetas_dict = {0:deepcopy(thetas)}
    weights_dict = {0:deepcopy(weights)}

    for i in range(niter_reg):
        for d in range(D):
            Xreg = np.array([(tl.unfold(x,d) @ tl.tenalg.khatri_rao([thetas[d_] for d_ in range(D) if d_ != d])).reshape(-1) for x in X])
            Sigma = Fmat.T @ Xreg
            thetas[d] = tl.vec_to_tensor(np.linalg.multi_dot([np.linalg.pinv(Sigma.T @ Sigma), Sigma.T, y]), shape=thetas[d].shape)
        thetas = [theta / np.linalg.norm(theta,axis=0) for theta in thetas]
        Thetas_unscaled = [tl.cp_to_tensor([np.ones(1),[theta[:,[r]] for theta in thetas]]) for r in range(rank)]
        XregW = np.array([[tl.tenalg.inner(Theta, x) for Theta in Thetas_unscaled] for x in X])
        SigmaW = Fmat.T @ XregW
        weights = np.linalg.multi_dot([np.linalg.pinv(SigmaW.T @ SigmaW), SigmaW.T, y])
        Theta = tl.cp_to_tensor([weights, thetas])
        delta = rewards - np.array([tl.tenalg.inner(x, Theta) for x in X])
        loss = 1/np.sqrt(N) * np.sqrt(np.linalg.multi_dot([delta.T, Fmat, Fmat.T, delta]))
        loss_dict[i+1] = loss
        thetas_dict[i+1] = deepcopy(thetas)
        weights_dict[i+1] = deepcopy(weights)
    
    return {'thetas':thetas_dict, 'weights':weights_dict, 'loss':loss_dict}

def policy(state, theta, nactions, eps_policy):
    opt_action = np.argmax([np.dot(phi_fn(state, a, nactions).reshape(-1),theta) for a in range(nactions)])
    return (1-2*eps_policy)*np.eye(nactions)[opt_action,:] + eps_policy*np.ones(nactions)
    
def value_mc(states,gamma,tol,Xi,alpha_dirs,theta,nactions,d1,d2,d3,eps_policy):
    vals = []
    n_obs_mc = math.ceil(math.log(tol,gamma))
    for state0 in states:
        state = state0
        val = sum([policy(state, theta, nactions, eps_policy)[a]*tl.tenalg.inner(phi_fn(state,a,nactions), Xi) for a in range(nactions)])
        for t in range(n_obs_mc):
            action = np.random.choice(nactions,p=policy(state,theta,nactions,eps_policy))
            reward = tl.tenalg.inner(phi_fn(state,action,nactions), Xi)
            val += gamma**(t+1)*reward
            state_ = np.random.dirichlet(alpha_dirs[action]).reshape(d1,d2,d3)
            state = state_
        vals.append(val)
    return np.array(vals)
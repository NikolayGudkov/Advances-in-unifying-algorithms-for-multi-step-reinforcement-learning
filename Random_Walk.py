# State-dependent version of the Q(\sigma) algorithm tested in the prediction task
# The 19-state Random Walk task from DeAsis et al.(2018)

import numpy as np
n_nodes, N_episodes, N_runs=19, 100, 100

N_models=3
n=3
alpha=0.4
beta=0.99   # the rate of decay of the degree of samling
gamma=1

Left_reward=-1
Right_reward=1
RMS_error=np.zeros((N_models,N_episodes))
V_CF=(Right_reward-Left_reward)*np.arange(n_nodes+2)/(n_nodes+1)+Left_reward

RMS_error_dynamic_episode=np.zeros((N_episodes,N_runs))
np.random.seed(1)

# Dynamic episode-dependent sigma
for run in range(N_runs):
        Q=np.ones((n_nodes+2,2))*0
        RMS=0
        sigma=1
        if (np.mod(run,10)==0): print("Model 1, ","run=",run)
        for i in range(N_episodes):
            S=np.array([],dtype=int)
            A=np.array([],dtype=int)
            TD_delta=np.array([])
            state=int((n_nodes+1)/2)
            S=np.append(S,state)
            action=np.power(-1,np.random.binomial(1,0.5))
            A=np.append(A,action)
            T=np.Inf
            t=0
            stop_tau=0
            while not stop_tau:
                if (t<T):
                    next_state=state+action
                    S=np.append(S,next_state)
                    if (next_state==0):
                        T=t+1
                        R=Left_reward
                        TD_delta=np.append(TD_delta,R-Q[state,int(action==1)])
                    elif (next_state==(n_nodes+1)):
                        T=t+1
                        R=Right_reward
                        TD_delta=np.append(TD_delta,R-Q[state,int(action==1)])
                    else:
                        R=0
                        V=(Q[next_state,0]+Q[next_state,1])/2
                        next_action=np.power(-1,np.random.binomial(1,0.5))
                        A=np.append(A,next_action)
                        TD_delta=np.append(TD_delta,R+gamma*(sigma*Q[next_state,int(next_action==1)]+(1-sigma)*V)-Q[state,int(action==1)])
                        action=next_action
                        state=next_state                     
                tau=t-n+1
                if (tau>=0):
                    E=1
                    G=Q[S[tau],int(A[tau]==1)]
                    for k in range(tau,min([t,T-1])+1):
                        G+=E*TD_delta[k]
                        if (k<T-1):
                            E*=gamma*((1-sigma)*0.5+sigma)                     
                    Q[S[tau],int(A[tau]==1)]+=alpha*(G-Q[S[tau],int(A[tau]==1)]) 
                if (tau==T-1): stop_tau=1
                t+=1
            sigma*=beta
            V_EST=(Q[:,0]+Q[:,1])/2
            RMS=np.power(np.mean(np.power(V_CF[1:-1]-V_EST[1:-1],2)),0.5)
            RMS_error_dynamic_episode[i,run]=RMS

# Dynamic state-dependent sigma
RMS_error_dynamic_state=np.zeros((N_episodes,N_runs))
np.random.seed(1)
for run in range(N_runs):
        if (np.mod(run,10)==0): print("Model 2, ", "run=",run)
        Q=np.ones((n_nodes+2,2))*0
        RMS=0
        sigma=np.ones(n_nodes+2)
        for i in range(N_episodes):
            S=np.array([],dtype=int)
            A=np.array([],dtype=int)
            TD_delta=np.array([])
            state=int((n_nodes+1)/2)
            S=np.append(S,state)
            action=np.power(-1,np.random.binomial(1,0.5))
            A=np.append(A,action)
            T=np.Inf
            t=0
            stop_tau=0
            while not stop_tau:
                if (t<T):
                    next_state=state+action
                    S=np.append(S,next_state)
                    if (next_state==0):
                        T=t+1
                        R=Left_reward
                        TD_delta=np.append(TD_delta,R-Q[state,int(action==1)])
                    elif (next_state==(n_nodes+1)):
                        T=t+1
                        R=Right_reward
                        TD_delta=np.append(TD_delta,R-Q[state,int(action==1)])
                    else:
                        R=0
                        V=(Q[next_state,0]+Q[next_state,1])/2
                        next_action=np.power(-1,np.random.binomial(1,0.5))
                        A=np.append(A,next_action)
                        TD_delta=np.append(TD_delta,R+gamma*(sigma[next_state]*Q[next_state,int(next_action==1)]+(1-sigma[next_state])*V)-Q[state,int(action==1)])
                        action=next_action
                        state=next_state
                        sigma[next_state]*=beta       
                tau=t-n+1
                if (tau>=0):
                    E=1
                    G=Q[S[tau],int(A[tau]==1)]
                    for k in range(tau,min([t,T-1])+1):
                        G+=E*TD_delta[k]
                        if (k<T-1):
                            E*=gamma*((1-sigma[S[k]])*0.5+sigma[S[k]])     
                    Q[S[tau],int(A[tau]==1)]+=alpha*(G-Q[S[tau],int(A[tau]==1)]) 
                if (tau==T-1): stop_tau=1
                t+=1
            V_EST=(Q[:,0]+Q[:,1])/2
            RMS=np.power(np.mean(np.power(V_CF[1:-1]-V_EST[1:-1],2)),0.5)
            RMS_error_dynamic_state[i,run]=RMS
                
RMS_error_dynamic_state_mean=np.mean(RMS_error_dynamic_state,axis=1)
RMS_error_dynamic_state_std=np.std(RMS_error_dynamic_state,axis=1)

RMS_error_dynamic_episode_mean=np.mean(RMS_error_dynamic_episode,axis=1)
RMS_error_dynamic_episode_std=np.std(RMS_error_dynamic_episode,axis=1)

print('Average standard deviation dynamic sigma(episode):', np.mean(RMS_error_dynamic_episode_std))
print('Average standard deviation dynamic sigma(state):',np.mean(RMS_error_dynamic_state_std))

import matplotlib.pyplot as plt
fig, ax= plt.subplots()
    
leg="Dynamic episode-dependent $\sigma$ "
ax.plot(range(N_episodes),RMS_error_dynamic_episode_mean, color='blue', label=leg)
#ax.plot(range(N_episodes),RMS_error_dynamic_episode_mean+1.96*RMS_error_dynamic_state_std, color='blue',linestyle=':')
#ax.plot(range(N_episodes),RMS_error_dynamic_episode_mean-1.96*RMS_error_dynamic_state_std, color='blue',linestyle=':')

leg="Dynamic state-dependent $\sigma$ "
ax.plot(range(N_episodes),RMS_error_dynamic_state_mean, color='red', label=leg)
#ax.plot(range(N_episodes),RMS_error_dynamic_state_mean+1.96*RMS_error_dynamic_state_std, color='red',linestyle=':')
#ax.plot(range(N_episodes),RMS_error_dynamic_state_mean-1.96*RMS_error_dynamic_state_std, color='red',linestyle=':')
ax.legend()
ax.set_ylim(0.0, 0.4)
ax.set_xlabel('Episodes')
ax.set_ylabel('Root-mean-square error')
ax.set_title('19-State Random Walk')
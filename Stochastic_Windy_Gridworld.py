# State-dependent version of the Q(\sigma) algorithm in the control task
# The Stochastic Windy Grid world from DeAsis et al.(2018)

import numpy as np

gamma, epsilon, N_x, N_y, N_a, Reward, N_episodes, N_runs=1, 0.1, 6, 9, 4, -1, 100, 100
i_start,j_start,i_end,j_end=3,0,3,7
wind=np.array([0,0,0,1,1,1,2,2,1,0])
actions=np.array([0,1,2,3])

state_start=np.array([i_start,j_start])
state_end=np.array([i_end,j_end])

N_models=5 # Includes static versions of the Q(\sigma) algorithm Q(0), Q(0.5), Q(1) and dynamic versions of the Q(\sigma) algorithm
alpha_param=np.arange(0.1,1.1,0.1)
n=3

# Define the stochastic Grid World Environment
def move(state,action):
    i=state[0]
    j=state[1]
    
    if (np.random.binomial(1,0.9)):
        #Shift by action
        if (action==0):
            i=np.max([i-1,0])
        if (action==1):
            j=np.min([j+1,N_y])
        if (action==2):
            i=np.min([i+1,N_x])
        if (action==3):
            j=np.max([j-1,0])
    else:
        #Random shift to 
        r_i=np.random.choice([-1,0,1])
        r_j=np.random.choice([-1,0,1])
        while (r_i==0 and r_j==0):
            r_i=np.random.choice([-1,0,1])
            r_j=np.random.choice([-1,0,1])
        i=np.max([np.min([i+r_i,N_x]),0])
        j=np.max([np.min([j+r_j,N_y]),0])
        
    #Shift by wind blow
    i=np.max([i-wind[j],0])
    return np.array([i,j])

def select_action(Q,state):
    # Simulate action using the epsilon-greedy policy
    if (np.random.binomial(1,epsilon)):
        return np.random.choice(actions)
    else:
        return np.random.choice(np.where(Q[state[0],state[1],:]==np.max(Q[state[0],state[1],:]))[0])
    
def expected_value(Q,state):
    # Find optimal actions
    optimal_actions=np.where(Q[state[0],state[1],:]==np.max(Q[state[0],state[1],:]))[0]
    probs=np.ones(len(actions))*epsilon/len(actions)
    probs[optimal_actions]+=(1-epsilon)/len(optimal_actions)
    return np.dot(Q[state[0],state[1],:],probs)
    
        
Average_reward=np.zeros((N_models,len(alpha_param),N_runs))
for i in range(3,N_models): 
    for j in range(len(alpha_param)):
        alpha=alpha_param[j]
        if (i==2 and j>6): continue
        if (i==3 and j>6): continue
        print('Model=',i,'alpha=',alpha)
        np.random.seed(1)
        for run in range(N_runs):    
            Average_reward_per_episode=0
            Q=np.zeros((N_x+1,N_y+1,N_a))
            if (i==0):
                sigma_param=0
                beta=1
            elif (i==1):
                sigma_param=0.5
                beta=1
            elif (i==2):
                sigma_param=1
                beta=1
            elif (i==3):
                sigma_param=1
                beta=0.99
            elif (i==4):
                sigma_param=np.ones((N_x+1,N_y+1))
                beta=0.99
                
            for episode in range(N_episodes):
                state=np.copy(state_start)
                action=select_action(Q,state)
                S=[state]
                A=np.array([action],dtype=int)
                TD_delta=np.array([])
                Total_reward=0
                T=np.Inf
                t=0
                stop_tau=0
                while not stop_tau:
                    if (t<T):
                        next_state=move(state,action)
                        S.append(next_state)
                        if (next_state[0]==state_end[0] and next_state[1]==state_end[1]):
                            T=t+1
                            R=0
                            TD_delta=np.append(TD_delta,R-Q[state[0],state[1],action])
                        else:
                            R=-1
                            V=expected_value(Q,next_state)
                            next_action=select_action(Q,next_state)
                            A=np.append(A,next_action)
                            if (i!=4):
                                sigma=sigma_param
                            else:
                                sigma=sigma_param[next_state[0],next_state[1]]
                                sigma_param[next_state[0],next_state[1]]*=beta
                            TD_delta=np.append(TD_delta,R+gamma*(sigma*Q[next_state[0],next_state[1],next_action]+(1-sigma)*V)-Q[state[0],state[1],action])
                            action=next_action
                            state=next_state
                            Total_reward+=R         
                    tau=t-n+1
                    if (tau>=0):
                        E=1
                        G=Q[S[tau][0],S[tau][1],A[tau]]
                        for k in range(tau,min([t,T-1])+1):
                            G+=E*TD_delta[k]
                            if (k<T-1):
                                if (i!=4):
                                    sigma=sigma_param
                                else:
                                    sigma=sigma_param[S[k][0],S[k][1]]
                                E*=gamma*((1-sigma)*0.5+sigma)
                        Q[S[tau][0],S[tau][1],A[tau]]+=alpha*(G-Q[S[tau][0],S[tau][1],A[tau]])             
                    if (tau==T-1): stop_tau=1
                    t+=1        
                if (i!=4): sigma_param*=beta      
                Average_reward_per_episode+=(Total_reward-Average_reward_per_episode)/(episode+1)
            Average_reward[i,j,run]=Average_reward_per_episode
    

import matplotlib.pyplot as plt
fig, ax= plt.subplots()

leg="Dynamic episode-dependent $\sigma$"
ax.plot(alpha_param[0:7],np.mean(Average_reward[3,0:8,],axis=1), color='blue', label=leg)
leg="Dynamic state-dependent $\sigma$"
ax.plot(alpha_param,np.mean(Average_reward[4,:],axis=1), color='red', label=leg)
ax.legend()
ax.set_ylim(-110, -60)
ax.set_xlabel('Step size')
ax.set_ylabel('Average Return per Episode')
ax.set_title('Stochastic Windy Gridworld')

print('Average standard deviation dynamic sigma(episode):', np.mean(np.std(Average_reward[3,0:8,],axis=1)))
print('Average standard deviation dynamic sigma(state):',np.mean(np.std(Average_reward[4,0:8,],axis=1)))



#leg="Q(0), Tree-backup"
#ax.plot(alpha_param,Average_reward[0,:], color='black', label=leg)
#leg="Q(0.5)"
#ax.plot(alpha_param,Average_reward[1,:], color='green', label=leg)
#leg="Q(1), Sarsa"
#ax.plot(alpha_param[0:7],Average_reward[2,0:7], color='orange', label=leg)

#%%
import gym

# %%
env = gym.make("FrozenLake-v1")
env.render()
env.close()


#%%
print(env.observation_space)
print(env.action_space)
print(env.observation_space.n)
print(env.action_space.n)


#%%
env.reset()
print(env.P[4][2])
print(env.P[14][1])
print(env.P[5][0])

env.render()

#%%
state = env.reset()
env.render()
print(env.step(1))
env.render()


#%%
state = env.reset()
print("Time step 0:")
env.render()

num_timesteps=20
for t in range(num_timesteps):
    random_action = env.action_space.sample()

    new_state, reward, done, _ = env.step(random_action)
    print(f"timestep {t+1}:")
    env.render()

    if done:
        break


#%%
import numpy as np
import gym
env=gym.make('FrozenLake-v1')

def value_evaluation(policy):
    num_iterations=50000000000000000000
    thre=1e-20
    gamma=0.9
    value_table=np.zeros(env.observation_space.n)
    
    for i in range(num_iterations):
        updated_value_table=np.copy(value_table)
        for s in range(env.observation_space.n):
            a=policy[s]
            value_table[s]=sum([prob*(r+gamma*updated_value_table[s_]) 
                          for prob,s_,r,_ in env.P[s][a]])
        if (np.sum(np.fabs(updated_value_table-value_table))<=thre):
                print("converged")
                break
    return value_table

def policy_improvement(value_table):
    gamma=0.9
    policy=np.ones(env.observation_space.n)
    
    for s in range(env.observation_space.n):
        q_values=[sum([prob*(r+gamma*value_table[s_]) 
                          for prob,s_,r,_ in env.P[s][a]])
                        for a in range(env.action_space.n)]
        policy[s]=np.argmax(np.array(q_values))
    return policy

def policy_iteration(env):
        num_iterations=30000000000000000000000
        policy=np.zeros(env.observation_space.n)
        
        for i in range(num_iterations):
            value_function=value_evaluation(policy)
            new_policy=policy_improvement(value_function)
            if (np.all(policy==new_policy)):
                print("optimal policy")
                break
            else:
                policy=new_policy
        return value_function, policy

optimal_value, optimal_policy = policy_iteration(env)
print(optimal_value)
print(optimal_policy)

obs = env.reset()
episode_reward = 0.0
for _ in range(30000):
    print(f"try: {_}")
    print(f"current_state: {obs}")
    action=np.int_(optimal_policy[obs])
    print(f"action: {action}")
    obs,reward,done,info=env.step(action)
    print(f"new_state: {obs}")
    # env.render()
    episode_reward+=reward
    print(f"reward: {episode_reward}")
    if done:
        print("goal")
        break
        print('rewards:', episode_reward)
        episode_reward=0.0
        obs=env.reset()

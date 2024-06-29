import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt

class Qlearn:
    
    def __init__(self, 
                 alpha, 
                 gamma, 
                 nbins):
        self.env = gym.make('CartPole-v0')
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1
        self.create_qtab_bins(nbins)
        
    def create_qtab_bins(self, nbins):  
        self.bins = [
		np.linspace(-4.8, 4.8, nbins),
		np.linspace(-4, 4, nbins),
		np.linspace(-.418, .418, nbins),
		np.linspace(-4, 4, nbins)]
        # self.bins = []
        # for i,j in zip(self.env.observation_space.low, self.env.observation_space.high):
        #     self.bins.append(np.linspace(i,j,nbins))
        self.q_table = np.zeros([nbins] * self.env.observation_space.shape[0] + [self.env.action_space.n]) #(np.random.random([nbins] * self.env.observation_space.shape[0] + [self.env.action_space.n])-1)/100
    
    def get_state_q(self, states):
        return tuple(np.digitize(x=states[i], bins=self.bins[i])-1 for i in range(len(states)))
    
    def run(self, max_run):
        reward = []
        for n in range(max_run):
            states = self.get_state_q(self.env.reset()[0])
            done = False

            steps = 0
            while not done:
                if np.random.random() > self.epsilon:
                    act = np.argmax(self.q_table[states])
                else:
                    act = np.random.randint(2)
                
                obs, r, term, trunc, info= self.env.step(act)

                # if trunc or (abs(obs[0]) > 2.4) or (abs(obs[2]) > np.radians(12)):
                #     term = True

                done = term or trunc

                new_states = self.get_state_q(obs)

                q_max = max(self.q_table[new_states])
                q_now = self.q_table[states + (act,)]

                # if term and (steps < 200):
                #     r = -375

                q_new = q_now + self.alpha * (r + self.gamma * q_max - q_now)
                self.q_table[states + (act,)] = q_new
                states = new_states
                steps += r
            if n>100:
                self.epsilon -= 0.9/max_run
            reward.append(steps)
        return reward

if __name__ == '__main__':
    temp = Qlearn(0.1, 0.95, 30)
    temp_rewards = temp.run(1000)
    plt.plot(temp_rewards)
    plt.show()


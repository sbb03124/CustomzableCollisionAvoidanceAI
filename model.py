import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import clone_model, Model


class Memory():
    def __init__(self, maxlen, window_length):
        self.window_length = window_length
        self.state0 = deque(maxlen=maxlen)
        self.state0_next = deque(maxlen=maxlen)
        self.state1 = deque(maxlen=maxlen)
        self.state1_next = deque(maxlen=maxlen)
        self.reward0 = deque(maxlen=maxlen)
        self.reward1 = deque(maxlen=maxlen)
        self.done = deque(maxlen=maxlen)
        self.action = deque(maxlen=maxlen)

    def append(self, s0, s0_next, s1, s1_next, r0, r1, d, a):
        if self.window_length is None:
            self.state0.append(s0)
            self.state0_next.append(s0_next)
            self.state1.append(s1)
            self.state1_next.append(s1_next)
            self.reward0.append(r0)
            self.reward1.append(r1)
            self.done.append(d)
            self.action.append(a)
        else:
            if len(self.done)>=1 and not self.done[-1]:
                state0 = list(self.state0[-1][1:]) + [s0]
                state0_next = list(self.state0_next[-1][1:]) + [s0_next]

                state1 = list(self.state1[-1][1:]) + [s1]
                state1_next = list(self.state1_next[-1][1:]) + [s1_next]

            else:
                # print('zeros_like')
                state0 = [np.zeros_like(s0) for _ in range(self.window_length-1)] + [s0]
                state0_next = [np.zeros_like(s0) for _ in range(self.window_length-2)] + [s0,s0_next]

                state1 = [np.zeros_like(s1) for _ in range(self.window_length-1)] + [s1]
                state1_next = [np.zeros_like(s1) for _ in range(self.window_length-2)] + [s1,s1_next]
            
            self.state0.append(np.array(state0))
            self.state0_next.append(np.array(state0_next))
            self.state1.append(np.array(state1))
            self.state1_next.append(np.array(state1_next))
            self.reward0.append(r0)
            self.reward1.append(r1)
            self.done.append(d)
            self.action.append(a)

    def get_recent_obs(self, state0, state1):
        if self.window_length is None:
            return state0, state1

        if len(self.done)>=1 and not self.done[-1]:
            state0 = list(self.state0[-1][1:]) + state0
            state1 = list(self.state1[-1][1:]) + state1
        else:
            # print('zeros_like')
            state0 = [np.zeros_like(state0) for _ in range(self.window_length-1)] + state0
            state1 = [np.zeros_like(state1) for _ in range(self.window_length-1)] + state1
        state0 = np.array(state0)
        state1 = np.array(state1)
        # print(state.shape)
        return state0, state1
    
    def get_sample_idx(self, num):
        idxs = np.random.randint(
            self.window_length-1, len(self.state0), num
        )
        for i in range(num):
            while self.done[idxs[i]-1] and self.done[idxs[i]]:
                idxs[i] =  np.random.randint(
                    self.window_length-1,
                    len(self.state0),
                )
        return idxs
    
    def __getitem__(self, idx):
        if type(idx) is int:
            return [
                self.state0[idx], self.state0_next[idx],
                self.state1[idx], self.state1_next[idx],
                self.reward0[idx], self.reward1[idx],
                self.done[idx], self.action[idx]
            ]
        else:
            return [
                [
                    self.state0[i], self.state0_next[i],
                    self.state1[i], self.state1_next[i],
                    self.reward0[i], self.reward1[i],
                    self.done[i], self.action[i]
                ] for i in idx
            ]

class OrnsteinUhlenbeckProcess():
    def __init__(self, theta, mu=0., sigma=1., dt=1e-3, size=1, sigma_min=None, n_steps_annealing=1000, nb_wormup=0):
        assert n_steps_annealing>nb_wormup
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.size = size

        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0
        self.nb_wormup = nb_wormup

        if sigma_min is not None:
            self.sigma_del = -float(sigma - sigma_min) / float(n_steps_annealing - nb_wormup)
            self.sigma_ini = sigma
            self.sigma_min = sigma_min
        else:
            self.sigma_del = 0.
            self.sigma_ini = sigma
            self.sigma_min = sigma

        self.noise_que = deque(maxlen=1000)

        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        self.noise_que.append(x)
        return x

    def reset_states(self):
        self.noise_que = deque(maxlen=1000)
        self.x_prev = np.random.normal(self.mu,self.current_sigma,self.size)

    @property
    def current_sigma(self):
        if self.n_steps<self.nb_wormup:
            sigma = self.sigma_ini
        else:
            sigma = max(self.sigma_min, self.sigma_del * float(self.n_steps-self.nb_wormup) + self.sigma_ini)
        return sigma

@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

class Agent():
    def __init__(self,
        actor_nn, critic_nn, actor_target_nn, critic_target_nn,
        random, memory,
        actor_optimizer, critic_optimizer,
        batch_size, gamma,
        target_model_update,
        grad_cliping,
        training=True,
        ):
        self.random = random
        self.memory = memory
        self.batch_size = batch_size
        self.gamma = gamma
        self.grad_cliping = grad_cliping
        self.training = training
        self.target_model_update = target_model_update
        # 暫定的な設定
        self.action_num = 1

        # mdel
        self.actor = actor_nn
        self.critic = critic_nn
        self.actor_target = actor_target_nn
        self.critic_target = critic_target_nn

        # 学習用の設定
        self.critic_opt = critic_optimizer
        self.actor_opt = actor_optimizer

        assert self.critic_opt!=self.actor_opt

    def train(self, train_actor=True, trian_critic=True,):
        sample_idx = self.memory.get_sample_idx(self.batch_size)
        
        exp = self.memory[sample_idx]

        shape2idx = []
        state0 = []
        state0_next = []
        state1 = []
        state1_next = []
        reward0 = []
        reward1 = []
        done = []
        action0 = []
        for n in range(self.batch_size):
            shape = np.array(exp[n][2]).shape
            oth_num = shape[1]
            # datanum = shape[0]*shape[2]
            datanum = shape[0]*10
            if shape not in shape2idx:
                shape2idx.append(shape)
                state0.append([])
                state0_next.append([])
                state1.append([])
                state1_next.append([])
                reward0.append([])
                reward1.append([])
                done.append([])
                action0.append([])
                
            idx = shape2idx.index(shape)
            #################################################
            state0[idx].append( exp[n][0].flatten() )
            state0_next[idx].append( exp[n][1].flatten() )
            state1[idx].append(
                np.concatenate( [ s for s in exp[n][2] ], axis=-1 ).rehsape((oth_num,datanum))
            )
            state1_next[idx].append(
                np.concatenate( [ s for s in exp[n][3] ], axis=-1 ).rehsape((oth_num,datanum))
            )
            reward0[idx].append( exp[n][4] )
            reward1[idx].append( exp[n][5] )
            done[idx].append( exp[n][6] )
            action0[idx].append( exp[n][7] )
            #################################################
            
        state0 = [tf.convert_to_tensor(np.array(s), dtype=np.float32) for s in state0 ]
        state0_next = [tf.convert_to_tensor(np.array(s), dtype=np.float32) for s in state0_next ]
        state1 = [tf.convert_to_tensor(np.array(s), dtype=np.float32) for s in state1 ]
        state1_next = [tf.convert_to_tensor(np.array(s), dtype=np.float32) for s in state1_next ]
        reward0 = [tf.convert_to_tensor(np.array(r).reshape((len(r),1)), dtype=np.float32) for r in reward0 ]
        reward1 = [tf.convert_to_tensor(np.array(r).reshape((len(r),1)), dtype=np.float32) for r in reward1 ]
        action0 = [tf.convert_to_tensor(np.array(a).reshape((len(a),1)), dtype=np.float32) for a in action0]
        # Trueが1なので，0になるように引き算
        done = [tf.convert_to_tensor((1 - np.array(d)).reshape((len(d),1)), dtype=np.float32) for d in done]
        
        losses = self.update(
            state0, state0_next, state1, state1_next, reward0, reward1, action0, done,
            train_actor=train_actor, trian_critic=trian_critic,
        )


        self.update_target(self.target_model_update)

        return losses

    @tf.function        
    def update_target(self, tau):
        update_target(
            self.actor_target.trainable_weights,
            self.actor.trainable_weights,
            tau
        )
        update_target(
            self.critic_target.trainable_weights,
            self.critic.trainable_weights,
            tau
        )
    
    def update(self, state0, state0_next, state1, state1_next, reward0, reward1, action0, done, train_actor=True, trian_critic=True,):
        if trian_critic:
            with tf.GradientTape() as tape:
                critic_loss = 0
                for s0, s0_next, s1, s1_next, r0, r1, a0, d in zip(
                    state0, state0_next, state1, state1_next, reward0, reward1, action0, done
                ):
                    target_reward0 = tf.stop_gradient(
                        tf.concat(
                            [
                                (
                                    r0  + d*self.gamma*self.critic_target(
                                        s0_next, s1_next, self.actor_target( s0, s1, gain=0.0 )
                                    )[:,0:1]
                                ),
                                (
                                    r1  + d*self.gamma*self.critic_target(
                                        s0_next, s1_next, self.actor_target( s0, s1, gain=1.0 )
                                    )[:,1:2]
                                )
                            ], axis=-1
                        )
                    )

                    pred = self.critic( s0, s1, a0 )

                    trian_critic = trian_critic + tf.sqrt(
                        tf.reduce_mean(
                            tf.square(
                                target_reward0 - pred
                            )
                        )
                    )
                
                # l2 normalize
                loss_reg_critic = 0
                for var in self.critic.trainable_variables:
                    if 'bias' not in var.name:
                        loss_reg_critic += tf.reduce_mean(tf.square(var))
                
                critic_loss += loss_reg_critic*1e-4
                

            critic_grad = tape.gradient(
                critic_loss, self.critic.trainable_variables
            )
            if self.grad_cliping is not None:
                critic_grad = [
                    None if gradient is None else tf.clip_by_norm(gradient, self.grad_clipping)
                    for gradient in critic_grad
                ]
            self.critic_opt.apply_gradients(
                zip(critic_grad, self.critic.trainable_variables)
            )

        else:
            critic_loss = None

        if train_actor:
            #update actor
            gain = np.random.uniform(low=0,high=1)
            
            actor_loss = 0
            for s0, s0_next, s1, s1_next, r0, r1, a0, d in zip(
                    state0, state0_next, state1, state1_next, reward0, reward1, action0, done
                ):
                
                pred_q = self.critic( s0, s1, self.actor(s0, s1, gain=gain) )
                actor_loss = actor_loss + ( - tf.reduce_mean( (1-gain)*pred_q[:,0:1] + gain*pred_q[:,1:2] ) )
                
            # l2 normalize
            loss_reg_actor = 0
            for var in self.actor.trainable_variables:
                if 'bias' not in var.name:
                    loss_reg_actor += tf.reduce_mean(tf.square(var))

            actor_grad = tape.gradient(
                actor_loss, self.actor.trainable_variables
            )
            if self.grad_cliping is not None:
                actor_grad = [
                    None if gradient is None else tf.clip_by_norm(gradient, self.grad_clipping)
                    for gradient in actor_grad
                ]

            self.actor_opt.apply_gradients(
                zip(actor_grad, self.actor.trainable_variables)
            )

        else:
            actor_loss = None

        return actor_loss, critic_loss

    def append(self, s0, s0_next, s1, s1_next, r0, r1, d, a):
        self.memory.append(
            s0, s0_next, s1, s1_next, r0, r1, d, a
        )

    def get_action(self, state0, state1, gain=None):
        inputs_ = self.memory.get_recent_obs(state0, state1)

        if gain is None:
            gain = np.random.uniform(low=0,high=1)

        act = self.actor(state0,state1,gain).numpy().flatten()

        if self.training:
            noise = self.random.sample()
            act += noise

        return noise
    
    def load_weights(self, fname):
        self.actor.load_weights(
            fname.replace('.', '_actor.')
        )
        self.critic.load_weights(
            fname.replace('.', '_critic.')
        )
    
    def save_weights(self, fname):
        self.actor.save_weights(
            fname.replace('.', '_actor.')
        )
        self.critic.save_weights(
            fname.replace('.', '_critic.')
        )


if __name__=='__main__':
    raise NotImplementedError
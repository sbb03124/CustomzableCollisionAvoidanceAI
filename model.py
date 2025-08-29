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

    def get_recent_obs(self, obs):
        if self.window_length is None:
            return obs
        state0, state1 = obs
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

    def train(self, train_actor=True, trian_critic=True, smoothing_gain=0):
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
        action = []
        oth_input_num = 10
        for n in range(self.batch_size):
            shape = np.array(exp[n][2]).shape
            if shape not in shape2idx:
                shape2idx.append(shape)
                state0.append([])
                state0_next.append([])
                state1.append([])
                state1_next.append([])
                reward0.append([])
                reward1.append([])
                done.append([])
                action.append([])
            idx = shape2idx.index(shape)
            if len(state0[idx])==0:
                state0[idx].append(
                    [exp[n][0][:,:3]]
                )
                state1[idx].append(
                    [exp[n][1][:,:3]]
                )
                if oth_num>0:
                    state0[idx].append(
                        [
                            [
                                exp[n][0][:,3+oth_idx*oth_input_num:3+(1+oth_idx)*oth_input_num].flatten()
                                for oth_idx in range(oth_num)
                            ]
                        ]
                    )
                    state1[idx].append(
                        [
                            [
                                exp[n][1][:,3+oth_idx*oth_input_num:3+(1+oth_idx)*oth_input_num].flatten()
                                for oth_idx in range(oth_num)
                            ]
                        ]
                    )
                else:
                    state0[idx].append(None)
                    state1[idx].append(None)
            else:
                oth_num = int( (shape[1]-3)/oth_input_num )
                state0[idx][0].append( exp[n][0][:,:3] )
                state1[idx][0].append( exp[n][1][:,:3] )
                if oth_num>0:
                    state0[idx][1].append(
                        [
                            exp[n][0][:,3+oth_idx*oth_input_num:3+(1+oth_idx)*oth_input_num].flatten()
                            for oth_idx in range(oth_num)
                        ]
                    )
                    state1[idx][1].append(
                        [
                            exp[n][1][:,3+oth_idx*oth_input_num:3+(1+oth_idx)*oth_input_num].flatten()
                            for oth_idx in range(oth_num)
                        ]
                    )
            rewards[idx].append(exp[n][2])
            done[idx].append(exp[n][3])
            action[idx].append(exp[n][4])

        state0 = [ [tf.convert_to_tensor(np.array(_s), dtype=np.float32) if _s is not None else None for _s in s ] for s in state0]
        state1 = [ [tf.convert_to_tensor(np.array(_s), dtype=np.float32) if _s is not None else None for _s in s ] for s in state1]
        rewards = [tf.convert_to_tensor(np.array(r).reshape((len(r),1)), dtype=np.float32) for r in rewards]
        action0 = [tf.convert_to_tensor(np.array(a).reshape((len(a),1)), dtype=np.float32) for a in action]
        # Trueが1なので，0になるように引き算
        done = [tf.convert_to_tensor((1 - np.array(d)).reshape((len(d),1)), dtype=np.float32) for d in done]
        
        action1 = [
            tf.convert_to_tensor(
                a.numpy().reshape((int(a.numpy().size/self.action_num),self.action_num)),
                dtype=np.float32
            )
            for a in  self.actor_target(
                [ s[0] for s in state1],
                [ s[1] for s in state1],
            )
        ]
        
        losses = self.update(
            state0, state1, rewards, done, action0, action1,
            train_actor=train_actor, trian_critic=trian_critic, smoothing_gain=smoothing_gain
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
    
    def update(self, state0, state1, reward, done, actions0, actions1, train_actor=True, trian_critic=True, smoothing_gain=0):
        raise NotImplementedError
        #update critic
        reward_concat = tf.concat(reward, axis=0)
        done_concat = tf.concat(done, axis=0)
        actions0_concat = tf.concat(actions0, axis=0)
        actions0_concat_noised = actions0_concat
        actions1_concat = tf.concat(actions1, axis=0)

        state0_input_wp = [ s[0] for s in state0 ]
        state0_input_oth = [ s[1] for s in state0 ]
        
        state1_input_wp = [ s[0] for s in state1 ]
        state1_input_oth = [ s[1] for s in state1 ]
        
        state0_input_wp_noise  = [ s[0] + tf.random.normal(s[0].shape,0,0.02) if s[0] is not None else None for s in state0  ]
        state0_input_oth_noise = [ s[1] + tf.random.normal(s[1].shape,0,0.02) if s[1] is not None else None for s in state0  ]
        actions0_noised = [ _a + tf.random.normal(_a.shape,0,0.02) for _a in actions0]
        # state1_input_wp_noise = [ s[0] + tf.random.normal(s[0].shape,0,0.05) for s in state1 ]
        # state1_input_oth_noise = [ s[1] + tf.random.normal(s[1].shape,0,0.05) for s in state1 ]
        
        with tf.GradientTape() as tape:
            target_reward = tf.stop_gradient(
                reward_concat + done_concat*self.gamma*self.critic_target(actions1, state1_input_wp, state1_input_oth, training=False)
            )
            critic_loss = self.critic_loss(
                self.critic(
                    # actions0,
                    actions0_noised,
                    state0_input_wp,
                    state0_input_oth,
                    training=True,
                ),
                target_reward
            )

            # l2 normalize
            loss_reg_critic = 0
            for var in self.critic.trainable_variables:
                if 'bias' not in var.name:
                    loss_reg_critic += tf.reduce_mean(tf.square(var))
            
            critic_loss += loss_reg_critic*1e-4

            Lt = tf.reduce_mean(
                tf.square(
                    tf.stop_gradient(
                        self.critic(
                            actions0,
                            state0_input_wp,
                            state0_input_oth,
                            training=True,
                        )
                    ) - self.critic(
                        actions0_noised,
                        state0_input_wp,
                        state0_input_oth,
                        training=True,
                    )
                )
            ) + tf.reduce_mean(
                tf.square(
                    tf.stop_gradient(
                        self.critic(
                            actions0,
                            state0_input_wp,
                            state0_input_oth,
                            training=True,
                        )
                    ) - self.critic(
                        actions1,
                        state1_input_wp,
                        state1_input_oth,
                        training=True,
                    )
                )
            )
            critic_loss += Lt*1e-2

        critic_grad = tape.gradient(
            critic_loss, self.critic.trainable_variables
        )
        if self.grad_cliping is not None:
            critic_grad = [
                None if gradient is None else tf.clip_by_norm(gradient, self.grad_clipping)
                for gradient in critic_grad
            ]
        if trian_critic:
            self.critic_opt.apply_gradients(
                zip(critic_grad, self.critic.trainable_variables)
            )
        
    
        #update actor
        with tf.GradientTape() as tape:
            # j_pi
            actor_out = self.actor(
                state0_input_wp,
                state0_input_oth,
                training=True
            )
            pred = self.critic(
                actor_out,
                state0_input_wp,
                state0_input_oth,
                training=False
            )
            loss_j_pi = -tf.reduce_mean(pred)

            # l2 normalize
            loss_reg_actor = 0
            for var in self.actor.trainable_variables:
                if 'bias' not in var.name:
                    loss_reg_actor += tf.reduce_mean(tf.square(var))
            
            actor_loss = loss_j_pi + loss_reg_actor*1e-3

            # smoothing
            next_ = self.actor(
                state1_input_wp,
                state1_input_oth,
            )
            noised_ = self.actor(
                state0_input_wp_noise,
                state0_input_oth_noise,
            )
            
            if type(actor_out) is list:
                Lt = tf.sqrt(
                    tf.reduce_mean(
                        tf.square(
                            tf.concat(actor_out, axis=0) - tf.concat(next_, axis=0)
                        )
                    )
                ) + tf.sqrt(
                    tf.reduce_mean(
                        tf.square(
                            tf.concat(actor_out, axis=0) - tf.concat(noised_, axis=0)
                        )
                    )
                )    
            else:
                Lt = tf.sqrt(tf.reduce_mean(tf.square(actor_out - next_))) + tf.sqrt(tf.reduce_mean(tf.square(actor_out - noised_)))

            actor_loss += Lt*1e-2
            
        
        actor_grad = tape.gradient(
            actor_loss, self.actor.trainable_variables
        )
        if self.grad_cliping is not None:
            actor_grad = [
                None if gradient is None else tf.clip_by_norm(gradient, self.grad_clipping)
                for gradient in actor_grad
            ]
        if train_actor:
            self.actor_opt.apply_gradients(
                zip(actor_grad, self.actor.trainable_variables)
            )


        self.grad_info = [[],[]]
        for name, grad in zip(self.actor.trainable_weights, actor_grad):
            self.grad_info[0] += ['(Actor)'+name.name+'_min', '(Actor)'+name.name+'_max', '(Actor)'+name.name+'_mean']
            self.grad_info[1] += [grad.numpy().min(), grad.numpy().max(), grad.numpy().mean()]
        for name, grad in zip(self.critic.trainable_weights, critic_grad):
            self.grad_info[0] += ['(Critic)'+name.name+'_min', '(Critic)'+name.name+'_max', '(Critic)'+name.name+'_mean']
            self.grad_info[1] += [grad.numpy().min(), grad.numpy().max(), grad.numpy().mean()]

        return actor_loss, critic_loss

    def append(self, s, s_next, r, d, a):
        raise NotImplementedError
        self.memory.append(
            s, s_next, r, d, a
        )

    def get_action(self, state, evalu=False, features=False, attention_score=False):
        raise NotImplementedError
        _in = self.memory.get_recent_obs(state)
        # act = self.actor.predict( [np.array([_in])] ).flatten()[0]
        oth_input_num = 10
        wp = np.array(_in)[:,:3]
        if int((_in.shape[1]-3)/oth_input_num) > 0:
            oth = np.array(
                [
                    np.array(_in)[:,3+oth_input_num*n:3+oth_input_num*(n+1)].flatten()
                    for n in range(int((_in.shape[1]-3)/oth_input_num))
                ]
            )
        else:
            oth = None
        if attention_score:
            act, alpha = self.actor(
                np.array([wp]),
                np.array([oth]) if oth is not None else None,
                attention_score = attention_score
            )
            act = act.numpy()[0]
            alpha = alpha.numpy()[0].flatten()
        else:
            act = self.actor(
                np.array([wp]),
                np.array([oth]) if oth is not None else None,
            ).numpy()[0]

        if self.training:
            noise = self.random.sample()
            act += noise
        act = np.clip(act, -1, 1)
        if evalu:
            evalu = self.critic.evalu(
                np.array([act]),
                np.array([wp]),
                np.array([oth]) if oth is not None else None,
            ).numpy()[0]

            state_actions = np.array(
                [
                    self.critic.evalu(
                        np.array([np.full(act.shape,_a)]),
                        np.array([wp]),
                        np.array([oth]) if oth is not None else None,
                    ).numpy()[0] for _a in np.linspace(-1,1,21)
                ]
            )
            if attention_score:
                return act, evalu, state_actions, alpha
            else:
                return act, evalu, state_actions

        else:
            return act
    
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
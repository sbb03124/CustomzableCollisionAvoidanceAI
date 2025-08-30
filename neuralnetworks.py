import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Lambda


class actor_nn(Model):
    def __init__(self,):
        super().__init__()

        self.encode_state0 = Sequential(
            Dense(256, activation='relu'),
            Dropout(0.1),
            Dense(256, activation='relu'),
            Dropout(0.1),
            Dense(256, activation='softsign'),
        )
        
        self.encode_state1 = Sequential(
            Dense(256, activation='relu'),
            Dropout(0.1),
            Dense(256, activation='relu'),
            Dropout(0.1),
            Lambda(lambda x: tf.reduce_sum(x,axis=1)),
            Dense(256, activation='softsign'),
        )

        self.actor_nn = Sequential(
            Dense(256, activation='relu'),
            Dropout(0.1),
            Dense(256, activation='relu'),
            Dropout(0.1),
            Dense(128, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='softsign'),
        )


        self.activation = Activation('relu')
        self.softsign = Activation('softsign')
        self.dropout = Dropout(0.1)

    def build(self, input_shapes):
        self.__call__(
            *[ np.random.random(size=shape) for shape in input_shapes]
        )

    def call(self, s0, s1, gain=0.5, training=False):
        
        s0_encode = self.encode_state0(s0,training=training)
        s1_encode = self.encode_state0(s1,training=training)

        encoded = s0_encode*(1-gain) + s1_encode*gain

        out = self.actor_nn(encoded,training=training)

        return out
    

class critic_nn(Model):
    def __init__(self,):
        super().__init__()

        self.critic_state0 = Sequential(
            Dense(256, activation='relu'),
            Dropout(0.1),
            Dense(256, activation='relu'),
            Dropout(0.1),
            Dense(128, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='linear'),
        )
        
        self.critic_state1 = Sequential(
            Dense(256, activation='relu'),
            Dropout(0.1),
            Dense(256, activation='relu'),
            Dropout(0.1),
            Lambda(lambda x: tf.reduce_sum(x,axis=1)),
            Dense(256, activation='relu'),
            Dropout(0.1),
            Dense(256, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='linear'),
        )


    def call(self, s0, s1, act, training=False):
        
        q_s0 = self.critic_state0(s0)
        q_s1 = self.critic_state1(s1)

        return tf.concat([q_s0,q_s1],axis=-1)
    
    def build(self, input_shapes):
        self.__call__(
            *[ np.random.random(size=shape) for shape in input_shapes]
        )
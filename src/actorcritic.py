import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
#import keras.backend as K
import tensorflow as tf

import random
from collections import deque



class ActorCritic:
    def __init__(self, env, sess):
        self.env = env
        self.sess = sess
        
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = .995
        self.gamma = .95
        self.tau = .125
        self.memory = deque(maxlen=2000)
        
        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()
        
        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.env.action_space.shape[0]])
        
        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output, actor_model_weights, -self.actor_critic_grad)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)
        
        self.critic_state_input, self.critic_action_input, self.critic_model = self.create_critic_model()
        _,_, self.target_critic_model = self.create_critic_model()
        self.critic_grads = tf.gradients(self.critic_model.output, self.critic_action_input)
        
        self.sess.run(tf.global_variables_initializer())
    
    
        
    # ========================================================================= #
    #                               Modell definieren                           #
    # ========================================================================= #
    
    def create_actor_model(self):
        state_input = Input(shape=self.env.observation_space.shape)
        h1 = Dense(24,activation='relu')(state_input)
        h2 = Dense(48,activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        output = Dense(self.env.action_space.shape[0], activation='relu')(h3)
        
        model = Model(input=state_input, output=output)
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        
        return state_input, model
    
    def create_critic_model(self):
        state_input = Input(shape=self.env.observation_space.shape)
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(48, activation='relu')(state_h1)
        
        action_input = Input(shape=self.env.action_space.shape)
        action_h1 = Dense(48)(action_input)
        
        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation='relu')(merged)
        
        output = Dense(1, activation='relu')(merged_h1)
        
        model= Model(input=[state_input, action_input], output=output)
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        
        return state_input, action_input, model
    
    
    
    # ========================================================================= #
    #                               Modell trainieren                           #
    # ========================================================================= #
    
    def train(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return
        
        rewards = []
        samples = random.sample(self.memory, batch_size)
        self._train_critic(samples)
        self._train_actor(samples)
    
    def _train_critic(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            if not done:
                target_action = self.target_actor_model.predict(new_state)
                future_reward = self.target_critic_model.predict([new_state, target_action])[0][0]
                reward += self.gamma * future_reward
                self.critic_model.fit([cur_state, action], reward, verbose=0)
    
    def _train_actor(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, _ = sample
            predicted_action = self.actor_model.predict(cur_state)
            grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input: cur_state,
                self.critic_action_input: predicted_action
            })[0]
            
            self.sess.run(self.optimize, feed_dict={
                self.actor_state_input: cur_state,
                self.actor_critic_grad: grads
            })
    
    def remember(self, cur_state, action, reward, new_state, done):
        self.memory.append([cur_state, action, reward, new_state, done])
    
    
    
    # ========================================================================= #
    #                               Ziel-Modell aktualisieren                   #
    # ========================================================================= #
    def _update_actor_target(self):
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_critic_model.get_weights()
        
        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]
        
        self.target_critic_model.set_weights(actor_target_weights)
    
    def _update_critic_target(self):
        critic_model_weights = self.critic_model.get_weights()
        critic_target_weights = self.critic_target_model.get_weights()
        
        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]
        
        self.critic_target_model.set_weights(critic_target_weights)
    
    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()
    
    
    # ========================================================================= #
    #                               Prediction                                  #
    # ========================================================================= #
    
    def act(self, cur_state):
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return self.actor_model.predict(cur_state)
    
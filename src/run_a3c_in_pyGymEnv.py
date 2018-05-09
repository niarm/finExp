import tensorflow as tf
import keras.backend as K
import gym
from actorcritic import ActorCritic

def main_endless():
    sess = tf.Session()
    K.set_session(sess)
    env = gym.make("Pendulum-v0")
    actor_critic = ActorCritic(env, sess)
        
    num_trials = 10000
    trial_len = 500
        
    cur_state = env.reset()
    action = env.action_space.sample()
        
    while True:
        env.render()
        cur_state = cur_state.reshape((1, env.observation_space.shape[0]))
        action = actor_critic.act(cur_state)
        action = action.reshape((1, env.action_space.shape[0]))
            
        new_state, reward, done, _ = env.step(action)
        new_state = new_state.reshape((1, env.observation_space.shape[0]))
            
        actor_critic.remember(cur_state, action, reward, new_state, done)
        actor_critic.train()
            
        cur_state = new_state


def main():
    sess = tf.Session()
    K.set_session(sess)
    env = gym.make("Pendulum-v0")
    actor_critic = ActorCritic(env, sess)
        
    num_trials = 10000
    trial_len = 500

    for trial in range(num_trials):
        print("Starting trial Nr.{}".format(trial))
        cur_state = env.reset()
        cur_state = cur_state.reshape((1, env.observation_space.shape[0]))
        
        for step in range(trial_len):
            if trial % 10 == 0:
                env.render()

            action = actor_critic.act(cur_state)
            action = action.reshape((1, env.action_space.shape[0]))

            new_state, reward, done, _ = env.step(action)
            new_state = new_state.reshape((1, env.observation_space.shape[0]))

            actor_critic.remember(cur_state, action, reward, new_state, done)
            actor_critic.train()

            cur_state = new_state
            if done:
                break


if __name__ == "__main__":
	main()
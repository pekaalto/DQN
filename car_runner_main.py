# get_ipython().magic(u'load_ext autoreload')
# get_ipython().magic(u'autoreload 2')

"""
Python 3.5, tensorflow 1.0.0

Trains first for train_episodes amount of episodes
and then starts playing with the best known policy with no exploration

Optionally save checkpoints to checkpoint_path (set checkpoint_path=None to not save anything)
Experience history is never saved

Training and playing can be early stopped by giving input (pressing enter in console)
"""

from dqn.agent import CarRacingDQN
import os
import tensorflow as tf
import gym
import _thread
import re
import sys

# SETTINGS

# to start training from scratch:
load_checkpoint = False
checkpoint_path = "data/checkpoint02"
train_episodes = float("inf")
save_freq_episodes = 400

# To play from existing checkpoint without any training:
# load_checkpoint = True
# checkpoint_path = "data/checkpoint01"
# train_episodes = 0 #or just give higher value to train the existing checkpoint more

model_config = dict(
    min_epsilon=0.1,
    max_negative_rewards=12,
    min_experience_size=int(1e4),
    num_frame_stack=3,
    frame_skip=3,
    train_freq=4,
    batchsize=64,
    epsilon_decay_steps=int(1e5),
    network_update_freq=int(1e3),
    experience_capacity=int(4e4),
    gamma=0.95
)

print(model_config)
########

env_name = "CarRacing-v0"
env = gym.make(env_name)

# tf.reset_default_graph()
dqn_agent = CarRacingDQN(env=env, **model_config)
dqn_agent.build_graph()
sess = tf.InteractiveSession()
dqn_agent.session = sess

saver = tf.train.Saver(max_to_keep=100)

if load_checkpoint:
    print("loading the latest checkpoint from %s" % checkpoint_path)
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    assert ckpt, "checkpoint path %s not found" % checkpoint_path
    global_counter = int(re.findall("-(\d+)$", ckpt.model_checkpoint_path)[0])
    saver.restore(sess, ckpt.model_checkpoint_path)
    dqn_agent.global_counter = global_counter
else:
    if checkpoint_path is not None:
        assert not os.path.exists(checkpoint_path), \
            "checkpoint path already exists but load_checkpoint is false"

    tf.global_variables_initializer().run()


def save_checkpoint():
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    p = os.path.join(checkpoint_path, "m.ckpt")
    saver.save(sess, p, dqn_agent.global_counter)
    print("saved to %s - %d" % (p, dqn_agent.global_counter))


def one_episode():
    reward, frames = dqn_agent.play_episode()
    print("episode: %d, reward: %f, length: %d, total steps: %d" %
          (dqn_agent.episode_counter, reward, frames, dqn_agent.global_counter))

    save_cond = (
        dqn_agent.episode_counter % save_freq_episodes == 0
        and checkpoint_path is not None
        and dqn_agent.do_training
    )
    if save_cond:
        save_checkpoint()


def input_thread(list):
    input("...enter to stop after current episode\n")
    list.append("OK")


def main_loop():
    """
    This just calls training function
    as long as we get input to stop
    """
    list = []
    _thread.start_new_thread(input_thread, (list,))
    while True:
        if list:
            break
        if dqn_agent.do_training and dqn_agent.episode_counter > train_episodes:
            break
        one_episode()

    print("done")


if train_episodes > 0:
    print("now training... you can early stop with enter...")
    print("##########")
    sys.stdout.flush()
    main_loop()
    save_checkpoint()
    print("ok training done")

sys.stdout.flush()

dqn_agent.max_neg_rewards = 100
dqn_agent.do_training = False

print("now just playing...")
print("##########")
sys.stdout.flush()
main_loop()

print("That's it. Good bye")

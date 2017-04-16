import numpy as np
from skimage import color, transform
import tensorflow.contrib.slim as slim
import tensorflow as tf
import itertools as it
from dqn.experience_history import ExperienceHistory

class DQN:
    """
    General DQN agent.
    Can be applied to any standard environment

    The implementation follows:
    Mnih et. al - Playing Atari with Deep Reinforcement Learning https://arxiv.org/pdf/1312.5602.pdf

    The q-network structure is different from the original paper

    see also:
    David Silver's RL course lecture 6: https://www.youtube.com/watch?v=UoPei5o4fps&t=1s
    """

    def __init__(self,
            env,
            batchsize=64,
            pic_size=(96, 96),
            num_frame_stack=4,
            gamma=0.95,
            frame_skip=1,
            train_freq=4,
            initial_epsilon=1.0,
            min_epsilon=0.1,
            render=True,
            epsilon_decay_steps=int(1e6),
            min_experience_size=int(1e3),
            experience_capacity=int(1e5),
            network_update_freq=5000,
            regularization = 1e-6,
            optimizer_params = None,
            action_map=None
    ):
        self.exp_history = ExperienceHistory(
            num_frame_stack,
            capacity=experience_capacity,
            pic_size=pic_size
        )

        # in playing mode we don't store the experience to agent history
        # but this cache is still needed to get the current frame stack
        self.playing_cache = ExperienceHistory(
            num_frame_stack,
            capacity=num_frame_stack * 5 + 10,
            pic_size=pic_size
        )

        if action_map is not None:
            self.dim_actions = len(action_map)
        else:
            self.dim_actions = env.action_space.n

        self.network_update_freq = network_update_freq
        self.action_map = action_map
        self.env = env
        self.batchsize = batchsize
        self.num_frame_stack = num_frame_stack
        self.gamma = gamma
        self.frame_skip = frame_skip
        self.train_freq = train_freq
        self.initial_epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.render = render
        self.min_experience_size = min_experience_size
        self.pic_size = pic_size
        self.regularization = regularization
        # These default magic values always work with Adam
        self.optimizer_params = optimizer_params or dict(learning_rate=0.0004, epsilon=1e-7)

        self.do_training = True
        self.playing_epsilon = 0.0
        self.session = None

        self.state_size = (self.num_frame_stack,) + self.pic_size
        self.global_counter = 0
        self.episode_counter = 0

    @staticmethod
    def process_image(img):
        return 2 * color.rgb2gray(transform.rescale(img[34:194], 0.5)) - 1

    def build_graph(self):
        input_dim_with_batch = (self.batchsize, self.num_frame_stack) + self.pic_size
        input_dim_general = (None, self.num_frame_stack) + self.pic_size

        self.input_prev_state = tf.placeholder(tf.float32, input_dim_general, "prev_state")
        self.input_next_state = tf.placeholder(tf.float32, input_dim_with_batch, "next_state")
        self.input_reward = tf.placeholder(tf.float32, self.batchsize, "reward")
        self.input_actions = tf.placeholder(tf.int32, self.batchsize, "actions")
        self.input_done_mask = tf.placeholder(tf.int32, self.batchsize, "done_mask")

        # These are the state action values for all states
        # The target Q-values come from the fixed network
        with tf.variable_scope("fixed"):
            qsa_targets = self.create_network(self.input_next_state, trainable=False)

        with tf.variable_scope("train"):
            qsa_estimates = self.create_network(self.input_prev_state, trainable=True)

        self.best_action = tf.argmax(qsa_estimates, axis=1)

        not_done = tf.cast(tf.logical_not(tf.cast(self.input_done_mask, "bool")), "float32")
        q_target = tf.reduce_max(qsa_targets, -1) * self.gamma * not_done + self.input_reward
        # select the chosen action from each row
        # in numpy this is qsa_estimates[range(batchsize), self.input_actions]
        action_slice = tf.stack([tf.range(0, self.batchsize), self.input_actions], axis=1)
        q_estimates_for_input_action = tf.gather_nd(qsa_estimates, action_slice)

        training_loss = tf.nn.l2_loss(q_target - q_estimates_for_input_action) / self.batchsize

        optimizer = tf.train.AdamOptimizer(**(self.optimizer_params))

        reg_loss = tf.add_n(tf.losses.get_regularization_losses())
        self.train_op = optimizer.minimize(reg_loss + training_loss)

        train_params = self.get_variables("train")
        fixed_params = self.get_variables("fixed")

        assert (len(train_params) == len(fixed_params))
        self.copy_network_ops = [tf.assign(fixed_v, train_v)
            for train_v, fixed_v in zip(train_params, fixed_params)]

    def get_variables(self, scope):
        vars = [t for t in tf.global_variables()
            if "%s/" % scope in t.name and "Adam" not in t.name]
        return sorted(vars, key=lambda v: v.name)

    def create_network(self, input, trainable):
        if trainable:
            wr = slim.l2_regularizer(self.regularization)
        else:
            wr = None

        # the input is stack of black and white frames.
        # put the stack in the place of channel (last in tf)
        input_t = tf.transpose(input, [0, 2, 3, 1])

        net = slim.conv2d(input_t, 8, (7, 7), data_format="NHWC",
            activation_fn=tf.nn.relu, stride=3, weights_regularizer=wr, trainable=trainable)
        net = slim.max_pool2d(net, 2, 2)
        net = slim.conv2d(net, 16, (3, 3), data_format="NHWC",
            activation_fn=tf.nn.relu, weights_regularizer=wr, trainable=trainable)
        net = slim.max_pool2d(net, 2, 2)
        net = slim.flatten(net)
        net = slim.fully_connected(net, 256, activation_fn=tf.nn.relu,
            weights_regularizer=wr, trainable=trainable)
        q_state_action_values = slim.fully_connected(net, self.dim_actions,
            activation_fn=None, weights_regularizer=wr, trainable=trainable)

        return q_state_action_values

    def check_early_stop(self, reward, totalreward):
        return False, 0.0

    def get_random_action(self):
        return np.random.choice(self.dim_actions)

    def get_epsilon(self):
        if not self.do_training:
            return self.playing_epsilon
        elif self.global_counter >= self.epsilon_decay_steps:
            return self.min_epsilon
        else:
            # linear decay
            r = 1.0 - self.global_counter / float(self.epsilon_decay_steps)
            return self.min_epsilon + (self.initial_epsilon - self.min_epsilon) * r

    def train(self):
        batch = self.exp_history.sample_mini_batch(self.batchsize)

        fd = {
            self.input_reward: "reward",
            self.input_prev_state: "prev_state",
            self.input_next_state: "next_state",
            self.input_actions: "actions",
            self.input_done_mask: "done_mask"
        }
        fd1 = {ph: batch[k] for ph, k in fd.items()}
        self.session.run([self.train_op], fd1)

    def play_episode(self):
        eh = (
            self.exp_history if self.do_training
            else self.playing_cache
        )
        total_reward = 0
        frames_in_episode = 0

        first_frame = self.env.reset()
        first_frame_pp = self.process_image(first_frame)

        eh.start_new_episode(first_frame_pp)

        while True:
            if np.random.rand() > self.get_epsilon():
                action_idx = self.session.run(
                    self.best_action,
                    {self.input_prev_state: eh.current_state()[np.newaxis, ...]}
                )[0]
            else:
                action_idx = self.get_random_action()

            if self.action_map is not None:
                action = self.action_map[action_idx]
            else:
                action = action_idx

            reward = 0
            for _ in range(self.frame_skip):
                observation, r, done, info = self.env.step(action)
                if self.render:
                    self.env.render()
                reward += r
                if done:
                    break

            early_done, punishment = self.check_early_stop(reward, total_reward)
            if early_done:
                reward += punishment

            done = done or early_done

            total_reward += reward
            frames_in_episode += 1

            eh.add_experience(self.process_image(observation), action_idx, done, reward)

            if self.do_training:
                self.global_counter += 1
                if self.global_counter % self.network_update_freq:
                    self.update_target_network()
                train_cond = (
                    self.exp_history.counter >= self.min_experience_size and
                    self.global_counter % self.train_freq == 0
                )
                if train_cond:
                    self.train()

            if done:
                if self.do_training:
                    self.episode_counter += 1

                return total_reward, frames_in_episode

    def update_target_network(self):
        self.session.run(self.copy_network_ops)


class CarRacingDQN(DQN):
    """
    CarRacing specifig part of the DQN-agent

    Some minor env-specifig tweaks but overall
    assumes very little knowledge from the environment
    """

    def __init__(self, max_negative_rewards=100, **kwargs):
        all_actions = np.array(
            [k for k in it.product([-1, 0, 1], [1, 0], [0.2, 0])]
        )
        # car racing env gives wrong pictures without render
        kwargs["render"] = True
        super().__init__(
            action_map=all_actions,
            pic_size=(96, 96),
            **kwargs
        )

        self.gas_actions = np.array([a[1] == 1 and a[2] == 0 for a in all_actions])
        self.break_actions = np.array([a[2] == 1 for a in all_actions])
        self.n_gas_actions = self.gas_actions.sum()
        self.neg_reward_counter = 0
        self.max_neg_rewards = max_negative_rewards

    @staticmethod
    def process_image(obs):
        return 2 * color.rgb2gray(obs) - 1.0

    def get_random_action(self):
        """
        Here random actions prefer gas to break
        otherwise the car can never go anywhere.
        """
        action_weights = 14.0 * self.gas_actions + 1.0
        action_weights /= np.sum(action_weights)

        return np.random.choice(self.dim_actions, p=action_weights)

    def check_early_stop(self, reward, totalreward):
        if reward < 0:
            self.neg_reward_counter += 1
            done = (self.neg_reward_counter > self.max_neg_rewards)

            if done and totalreward <= 500:
                punishment = -20.0
            else:
                punishment = 0.0
            if done:
                self.neg_reward_counter = 0

            return done, punishment
        else:
            self.neg_reward_counter = 0
            return False, 0.0

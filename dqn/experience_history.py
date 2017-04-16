import numpy as np


class ExperienceHistory:
    """
    This saves the agent's experience in windowed cache.
    Each frame is saved only once but state is stack of num_frame_stack frames

    In the beginning of an episode the frame-stack is padded
    with the beginning frame
    """

    def __init__(self,
            num_frame_stack=4,
            capacity=int(1e5),
            pic_size=(96, 96)
    ):
        self.num_frame_stack = num_frame_stack
        self.capacity = capacity
        self.pic_size = pic_size
        self.counter = 0
        self.frame_window = None
        self.init_caches()
        self.expecting_new_episode = True

    def add_experience(self, frame, action, done, reward):
        assert self.frame_window is not None, "start episode first"
        self.counter += 1
        frame_idx = self.counter % self.max_frame_cache
        exp_idx = (self.counter - 1) % self.capacity

        self.prev_states[exp_idx] = self.frame_window
        self.frame_window = np.append(self.frame_window[1:], frame_idx)
        self.next_states[exp_idx] = self.frame_window
        self.actions[exp_idx] = action
        self.is_done[exp_idx] = done
        self.frames[frame_idx] = frame
        self.rewards[exp_idx] = reward
        if done:
            self.expecting_new_episode = True

    def start_new_episode(self, frame):
        # it should be okay not to increment counter here
        # because episode ending frames are not used
        assert self.expecting_new_episode, "previous episode didn't end yet"
        frame_idx = self.counter % self.max_frame_cache
        self.frame_window = np.repeat(frame_idx, self.num_frame_stack)
        self.frames[frame_idx] = frame
        self.expecting_new_episode = False

    def sample_mini_batch(self, n):
        count = min(self.capacity, self.counter)
        batchidx = np.random.randint(count, size=n)

        prev_frames = self.frames[self.prev_states[batchidx]]
        next_frames = self.frames[self.next_states[batchidx]]

        return {
            "reward": self.rewards[batchidx],
            "prev_state": prev_frames,
            "next_state": next_frames,
            "actions": self.actions[batchidx],
            "done_mask": self.is_done[batchidx]
        }

    def current_state(self):
        # assert not self.expecting_new_episode, "start new episode first"'
        assert self.frame_window is not None, "do something first"
        return self.frames[self.frame_window]

    def init_caches(self):
        self.rewards = np.zeros(self.capacity, dtype="float32")
        self.prev_states = -np.ones((self.capacity, self.num_frame_stack),
            dtype="int32")
        self.next_states = -np.ones((self.capacity, self.num_frame_stack),
            dtype="int32")
        self.is_done = -np.ones(self.capacity, "int32")
        self.actions = -np.ones(self.capacity, dtype="int32")

        # lazy to think how big is the smallest possible number. At least this is big enough
        self.max_frame_cache = self.capacity + 2 * self.num_frame_stack + 1
        self.frames = -np.ones((self.max_frame_cache,) + self.pic_size, dtype="float32")

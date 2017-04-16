#get_ipython().magic(u'load_ext autoreload')
#get_ipython().magic(u'autoreload 2')

import sys
import os
sys.path.append(os.getcwd())
from unittest import TestCase
from dqn.experience_history import ExperienceHistory
import numpy as np

num_frame_stack = 3
size = 10
pic_size = (2, 2)

class TestExperienceHistory(TestCase):
    def test_add_frame(self):
        h = ExperienceHistory(
            num_frame_stack=num_frame_stack,
            capacity=size,
            pic_size=pic_size
        )

        #can't do anything because no episode started
        with self.assertRaises(AssertionError):
            h.current_state()
        with self.assertRaises(AssertionError):
            h.add_experience(None, None, None, None)

        frames = np.random.rand(4, 2, 2).astype("float32")

        # add the beginning frame
        h.start_new_episode(frames[0])

        # Check that padding works correctly
        assert (h.current_state() == frames[0]).all()
        assert (h.current_state().shape == (num_frame_stack,) + pic_size)

        # Now add next frame.
        # The action is action taken before this frame
        # and reward is the reward observed for this action
        # done is a flag if we ended in terminal state
        h.add_experience(frames[1], 4, False, 1.0)

        assert (h.current_state()[:2] == frames[0]).all()
        assert (h.current_state()[2] == frames[1]).all()
        assert (h.current_state().shape == (num_frame_stack,) + pic_size)

        # add one more experience and set episode as finished
        h.add_experience(frames[2], 5, True, 2.0)

        # now there should not be any padding for current state
        assert (h.current_state() == frames[:3]).all()
        assert (h.current_state().shape == (num_frame_stack,) + pic_size)

        assert np.all(h.next_states[:3] == np.array([[0, 0, 1], [0, 1, 2], [-1, -1, -1]]))
        assert np.all(h.prev_states[:3] == np.array([[0, 0, 0], [0, 0, 1], [-1, -1, -1]]))

        h.start_new_episode(frames[3])

        assert (h.current_state() == frames[3]).all()
        assert (h.current_state().shape == (num_frame_stack,) + pic_size)

        batch = h.sample_mini_batch(20)

        # Check that we don't sample from the part which is not yet written
        # i.e shouldn't see zeros (the caches are initialized with zeros)
        assert np.all(np.in1d(batch["reward"], [1., 2.]))
        assert np.all(np.in1d(batch["actions"], [4., 5.]))

        # when we arrived to 2nd frame was the only time when episode was not done
        dm = ~batch["done_mask"].astype(bool)
        assert np.all(batch["next_state"][dm] == np.array(frames[[0, 0, 1]]))

        # frames[2] in the history is overwritten by frames[3] because new episode has started,
        # however it doesn't matter because the terminal state shouldn't be used anywhere.
        assert np.all(batch["next_state"][~dm] == np.array(frames[[0, 1, 3]]))
        assert np.all((batch["prev_state"] == frames[0]) | (batch["prev_state"] == frames[1]))

    def test_many_frames(self):
        n_frames = 1000
        size = 30
        frames = np.ones((n_frames, 2, 2)).astype("float32") * np.arange(n_frames).reshape(-1, 1, 1)
        start_frame = np.ones((2, 2), "float32") * 10000
        h = ExperienceHistory(
            num_frame_stack=num_frame_stack,
            capacity=30,
            pic_size=pic_size
        )

        h.start_new_episode(start_frame)

        #add 10 frames
        for f in frames[:10]:
            h.add_experience(f, 12, False, 5.0)

        this_state = h.current_state()
        h.add_experience(frames[10], 10, False, 4)

        def a():
            assert np.all(this_state == frames[7:10])
            assert h.rewards[10] == 4
            assert h.actions[10] == 10
            assert not h.is_done[10]
            assert np.all(h.frames[h.prev_states[10]] == frames[7:10])
            assert np.all(h.frames[h.next_states[10]] == frames[8:11])

        # Check that adding one frame
        # doesn't mess up the existing history
        a()

        # add 29 more experiences and check that
        # the past experience is not changed
        for f in frames[11:40]:
            done = np.random.rand() > 0.5
            h.add_experience(f, 0, done, 1.0)
            if done:
                h.start_new_episode(start_frame)
            a()

        # adding one more experience should
        # overwrite the oldest experience:
        h.add_experience(frames[40], 1, False, 2.0)
        assert h.rewards[10] == 2.0
        assert h.actions[10] == 1
        with self.assertRaises(AssertionError):
            a()

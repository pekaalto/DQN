![](https://media.giphy.com/media/3og0IEKu84Ros9izyU/giphy.gif)


## Info
This project implements the DQN reinforcement learning agent similar to
[Human-level control through deep reinforcement
learning](http://www.davidqiu.com:8888/research/nature14236.pdf)

(See also David Silvers RL course [lecture 6](https://www.youtube.com/watch?v=UoPei5o4fps). This stuff is clearly and shortly explained in 1h15min onwards) 

The agent is applied to the Open AI gym's [2d-car-racing environment](https://gym.openai.com/envs/CarRacing-v0)

The structure of the q-network differs from the original paper.
In particular, the network here is much smaller and can be easily trained without GPU.
(It's easy to specify any other structures as well)

The agent learns to drive the car from pixels in a few hours and doesn't need any hand-crafted features.
There are some minor environment specific tweaks for the car-racing but the base-agent doesn't know about car-racing.

## Pre-trained agent
The checkpoint provided in the repo used the default parameters
specified in the runner/agent and 150000~ playing steps for learning.

The training took about 5h with CPU.
This agent is playing in the above gif and in this video:
https://youtu.be/CVZQOAlQib0

The agent sometimes cuts corners but other than that it can drive flawlessly for minutes.
There are some occasional mistakes though.

## Running instuctions
Just clone the repo and use car_runner_main.py .
The settings are specified in the beginning of the runner.

You can either train from scratch or load the existing checkpoint
from this repo and see the agent driving somewhat properly right away.
Or you can train the provided checkpoint more.

## Requirements
- Python 3.5 (will not work with python 2)
- OpenAI Gym (the car-racing environment)
- Tensorflow 1.0.0
- numpy
- scikit-image

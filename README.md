# Banana navigation project from Udacity deep RL nanodegree.
In this project we train an agent to navigate a 2D space collecting yellow bananas and avoiding blue bananas.

## Setup
First, we need to install the necessary Python packages and set up the environment. Follow steps 1-6. from [this](https://github.com/udacity/deep-reinforcement-learning#dependencies) github page. During step 3., make sure to install the required environments by running:
```
pip install gym[classic_control]
pip install gym[box2d]
```

Second, you may or may not need to install some of the packages manually. In my case, in addition to following the instructions, I had to run:
```
pip install torch
```
and
```
pip install unityagents
```
Also, make sure to install pandas:
```
pip install pandas
```

Third, we need to download the environment:
* Linux: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* Mac OS: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* Windows (32-bit): [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* Windows (64-bit): [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

## Using this repository
The main code is contained in notebook [Navigation_Solution.ipynb](./Navigation_Solution.ipynb).[^1] The code for the agend is in [agent.py](./agent.py) and the policy model is defined in [model.py](./model.py). File [checkpoint.pth](./checkpoint.pth) contains the weights corresponding to a trained agent. See [report.md](./report.md) for an outline of the implementation, training statistics and ideas for further improvements.


## Problem description
In this problem the agent moves along a 2D plane using one of the four actions at each step:
- `0` - walk forward 
- `1` - walk backward
- `2` - turn left
- `3` - turn right

[^1]: Keep in mind that you may not always be able to restart the environment after closing it. In those cases, restarting the kernel should help.


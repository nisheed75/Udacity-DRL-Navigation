# Deep Reinforcement Learning : Navigation

This project repository is my implementation for Project 1: Navigation for the Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)

## Project's Description 
For this project we have to train an agent to navigate a large square world and collect yellow bananas. The world contains both yellow and blue banana as depicted in the animated gif below.
![In Project 1, train an agent to navigate a large world.](images/banana.gif)

### Rewards:
1. The agent is given a reward of +1 for collecting a yellow banana
1. Reward of -1 for collecting a blue banana.

### State Space 
Has 37 dimensions and the contains the agents velocity, along with ray-based precpetion of objects around the agents foward direction.

### Actions 
Four discrete actions are available, corresponding to:

- 0 - move forward.
- 1 - move backward.
- 2 - turn left.
- 3 - turn right.


## Project's goal
The goal for the project is for the to collect as many yellow bananas as possible while avoiding blue bananas. The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.


### Quick Start Guide to Deep Reinforcement Learning
To understand this implementation you will have to have some understanding of Deep Reinforcement Learning. I provide write-up of my implementation in the <insert report.pdf>. However i recommend reading > [Reinforcement learning](https://skymind.ai/wiki/deep-reinforcement-learning).

At a high-level Deep Reinformcement Learning is the science behind developing goal-oriented algorithms, which learn how to attain a complex goal from a blank slate. [AlphaGo](https://deepmind.com/blog/alphago-zero-learning-scratch/)  is a famous example of how Deep Reinforcement Learning achieve superhuman performance and defeated the world champion. 

This project implement a Value Based method called [Deep Q-Networks](https://deepmind.com/research/dqn/)

## The Environment

The environment is based on [Unity ML-agents](https://github.com/Unity-Technologies/ml-agents)

Note: The project environment for this pr4oject is similar to, but not identical to the [Banana Collector](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector) environment on the Unity ML-Agents GitHub page.


### Exploring the Environment 

#### Step 1: Clone the DRLND Repository
1. Configure your Python environment by following [instructions in the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies). These instructions can be found in the [Readme.md](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Readme.md)
1. By following the instructions you will have PyTorch, the ML-Agents toolkits, and all the Python packages required to complete the project.
1. (For Windows users) The ML-Agents toolkit supports Windows 10. It has not been test on older version but it may work.

#### Step 2: Download the Unity Environment 
- For this projects you will need to install the Unity environment as described in the [Getting Started section](https://github.com/udacity/deep-reinforcement-learning/blob/master/p1_navigation/README.md) (The Unity ML-agant environment is already configured by Udacity)

  - Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
	Then, place the file in the p1_navigation/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

#### Step 3: Explore the Environment
After you have followed the instructions above, open Navigation.ipynb (located in the p1_navigation/ folder in the DRLND GitHub repository) and follow the instructions to learn how to use the Python API to control the agent.
    
#### (Optional) Build your Own Environment
For this project, we have built the Unity environment for you, and you must use the environment files that we have provided.

If you are interested in learning to build your own Unity environments after completing the project, you are encouraged to follow the instructions [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Getting-Started-with-Balance-Ball.md), which walk you through all of the details of building an environment from a Unity scene.

### Train a agent
There are 2 options for training the Agent:
1. Execute the provided notebook within this Nanodegree Udacity Online Workspace for "project #1  Navigation".
1. Or build your own local environment and make necessary adjustements for the path to the UnityEnvironment in the code.

Note: that the Workspace does not allow you to see the simulator of the environment; so, if you want to watch the agent while it is training, you should train locally.

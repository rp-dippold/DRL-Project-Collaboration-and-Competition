# DRL-Project-Collaboration-and-Competition

## Project Details
This project is based on a Unity environment to design, train, and evaluate deep reinforcement learning algorithms.
The environment used in this project is the Tennis environment.

<p align="center">
 <img src="tennis.png"/>
    <br>
    <em><b>Unity ML-Agents Tennis Environment</b></em>
</p>

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play..

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
* This yields a single **score** for each episode.

TThe environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

## Getting Started

### Project File Structure
The project is structured as follows:

📦project<br>
 ┣ 📂Tennis_Linux  **`(contains the Tennis environment for Linux based systems)`** <br>
 ┣ 📂Tennis_Windows_x86_64_1  **`(contains the Tennis environment for Windows 64-bit based systems)`** <br>
 ┣ 📂models  **`(contains the actors and critics states of successfully trained agents)`** <br>
 ┃ ┣ checkpoint_actor_0.pth<br>
 ┃ ┣ checkpoint_critic_0.pth<br>
 ┃ ┣ checkpoint_actor_1.pth<br>
 ┃ ┣ checkpoint_critic_1.pth<br>
 ┃ ┗ ... <br>
 ┣ 📂python **`(files required to set up the environment)`** <br>
 ┣ 📂score_plots **`(contains the score plots of successfully trained agents)`** <br>
 ┃ ┣ score_plot_1.png<br>
 ┃ ┣ training_output_1.png<br>
 ┃ ┗ ...<br>
 ┣ .gitignore <br>
 ┣ config.py  <br>
 ┣ config.yml <br>
 ┣ ddpg_agent.py **`(Unity agent for tennis environment)`**<br> 
 ┣ main.py **`(Python script to run trained agents or to train new ones)`**<br>
 ┣ model.py **`(Actor and Critic networks)`**<br>
 ┣ tennis.png <br>
 ┣ README.md <br>
 ┗ Report.md <br>
 
### Installation and Dependencies

The code of this project was tested on Linux (Ubuntu 20.04) and Windows 11. To get the code running on your local system, follow these steps which are base on Anaconda and pip:

1.  `conda create --name tennis python=3.8 -c conda-forge`
2.  `conda activate tennis`
3.  Create a directory where you want to save this project
4.  `git clone https://github.com/rp-dippold/DRL-Project-Collaboration-and-Competition.git`
5.  `cd python`
6.  `pip install .`
7.  `python -m ipykernel install --user --name tennis --display-name "tennis"`
8.  Install Pytorch:
    * [CPU]: `pip install torch torchvision torchaudio`
    * [GPU]: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117`.\
    Depending on your GPU and cudnn version a different pytorch version my be required. Please refer to 
    https://pytorch.org/get-started/locally/.


## Instructions
To run the code go into the directory where you installed the repository. First of all, open the file `config.yml` and check if `tennis_env` refers to the correct tennis environment. Set the tennis environment as follows:

* **`Windows 11`**: "./Tennis_Windows_x86_64/Tennis.exe"
* **`Linux`**: "./Tennis_Linux/Tennis.x86_64"

**Note that the first start of the Unity environment &mdash; as described below  &mdash; may take up to 30 seconds; all following start times are much shorter. So please be patient!**

#### Training Agents
Before training agents you should adapt the respective hyperparameters in config.yml. The current values allow to train agents that can reach an average score of +0.6 over 100 consecutive episodes.

To start training just enter the following command: `python main.py train`

If you want to watch the agents during training enter: `python main.py train --watch`

At the end of the training a window pops up that shows the scores and average scores over 100 consecutive episodes of
the agents for each episode. After closing this window, the program stops.

If the agents were trained successfully their weights are saved in the root directory and not in directory `models`.
The filenames are: `checkpoint_actor_0.pth` and `checkpoint_critic_0.pth` for the first agent and `checkpoint_actor_1.pth` and `checkpoint_critic_1.pth` for the second agent.

#### Running the Environment with Smart Agents
The following command runs the environment:

`python main.py run --actor_1_params <path to stored weights of 1st actor> --critic_1_params <path to stored weights of 1st critic> --actor_1_params <path to stored weights of 2nd actor> --critic_1_params <path to stored weights of 2nd critic>`

`<path to stored weights ...>` is the path to the directory plus the name of the `checkpoint_xxx.pth` file, e.g.
`./models/checkpoint_actor_0.pth`.

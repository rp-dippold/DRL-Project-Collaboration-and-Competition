import sys
import argparse
import torch
import time
import numpy as np
import matplotlib.pyplot as plt

from config import Config
from collections import deque
from ddpg_agent import Agent
from ddpg_agent import ReplayBuffer
from unityagents import UnityEnvironment
from unityagents.exception import UnityActionException

def train_agent(watch=False):
    """Train agents for an environment.

    If argument `watch` is set to 'True', the user can watch the agents during
    training. Default is 'False'. If the training was completed successfully a
    plot with the scores for each episode and the average scores over the last
    100 episodes is displayed in a separate window.

    Parameters
    ----------
    watch : bool (Default False)
        Shows environment during training if set to 'True'.

    Returns
    -------
        A list of the scores of all episodes collected during training and a
        list of average scores over the last 100 episodes.
        If the environment was solved by the agents, their parameters
        (state dictionaries) are saved in the current directory.
    """
    start_time = time.time()
    config = Config.get_config()

    # Determine if GPU is available and requested by the configuration
    if config.device == 'GPU' and torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    # Collect information regarding the environment and the agents (brain)
    env = UnityEnvironment(file_name=config.tennis_env, no_graphics=not watch)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size
    # Initialize environment and get number of agents and state size
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = env._n_agents[brain_name]
    state_size = env_info.vector_observations.shape[1]
    
    # Initialize agents
    agents = [Agent(state_size=state_size,
              action_size=action_size,
              add_noise=True) for _ in range(num_agents)]
    
    # Replay buffer to collect interactions of the agents with the 
    # environment for training.
    replay_buffer = ReplayBuffer(action_size, config.buffer_size,
                                 config.batch_size, config.random_seed, device)  
    
    # List containing scores from each episode
    scores_list = []
    avg_scores_list = []
    # Use a noise factor to increasingly lower the noise (exploration)
    # during training.
    noise = config.noise_start
    noise_reduce = config.noise_reduce
    noise_min = config.noise_min
    # Queue containing the amount of scores specified by scores_window
    scores_window = deque(maxlen=config.scores_window)
    for i_episode in range(1, config.n_episodes+1):
        # Reset the environment and agents' noise
        env_info = env.reset(train_mode=True)[brain_name]
        [agent.reset() for agent in agents]
        # Obtain state of the initial environment
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        # Play the game until it terminates (done) or at most max timesteps
        for _ in range(config.max_timesteps):
            # Get the action for the current state (for each agent)
            actions = np.vstack([agent.act(state.reshape(1, -1), noise) \
                                 for agent, state in zip(agents, states)])

            # Take the next step using the received actions (for each agent)
            env_info = env.step(actions)[brain_name]
            # Extract the next state from the environment (for each agent)
            next_states = env_info.vector_observations
            # Get the reward for the last actions (for each agent)
            rewards = env_info.rewards
            # Find out if the game is over (True) or still running (False)
            dones = env_info.local_done

            # Save experience in replay buffer
            for state, action, reward, next_state, done in \
                zip(states, actions, rewards, next_states, dones):
                replay_buffer.add(state, action, reward, next_state, done)
            # Allow the agents to learn from last results.
            for agent in agents:
                agent.step(replay_buffer)

            # Roll over the state to next time step and update score
            # (for each agent)
            states = next_states
            scores += rewards
            
            # Exit loop if episode has finished
            if np.any(dones):
                break 

        # Reduce noise factor
        noise = max(noise_min, noise*noise_reduce)
        # Save most recent score
        score = np.max(scores)
        scores_window.append(score)
        scores_list.append(score)
        avg_scores_list.append(np.mean(scores_window))
        
        # Print result of the episode on the terminal screen
        print_newline_after = 100
        avg_score = np.mean(scores_window)
        print(f'\rEpisode {i_episode}\tAverage Score: {avg_score:.5f}', end="")
        if i_episode % print_newline_after == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {avg_score:.5f}')
        if avg_score >= config.total_avg_reward:
            print(f'\nEnvironment solved in {i_episode} episodes!', end="")
            print(f'\t Average Score {avg_score:.5f}')
            print(f'Training took {(time.time() - start_time)/60:.2f} minutes.')
            # Save agent parameters
            for i, agent in enumerate(agents):
                torch.save(agent.actor_local.state_dict(), 
                           f'checkpoint_actor_{i}.pth')
                torch.save(agent.critic_local.state_dict(), 
                           f'checkpoint_critic_{i}.pth')
            break
    env.close()
    return scores_list, avg_scores_list

def run_model(actor_1_params_file, critic_1_params_file,
              actor_2_params_file, critic_2_params_file):
    """Runs the agents for an environment and prints out the final score.

    The tennis environment with two agents is started and it runs for one 
    episode. After its completion the final score of each agent and the 
    average over both agents is printed out.

    Parameters
    ----------
    actor_1_params_file : string
        Name of the file that contains the first actor's parameters.
    critic_1_params_file : string
        Name of the file that contains the first critic's parameters.
    actor_2_params_file : string
        Name of the file that containes the second actor's parameters.
    critic_2_params_file : string
        Name of the file that contains the second critic's parameters.
    """
    config = Config.get_config()
    env = UnityEnvironment(file_name=config.tennis_env)
    # Get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # Initialize environment
    env_info = env.reset(train_mode=True)[brain_name]
    # Number of states and actions and agents
    state_size = env_info.vector_observations.shape[1]
    action_size = brain.vector_action_space_size
    num_agents = env._n_agents[brain_name]
    # Create default (untrained) agents
    agents = [Agent(state_size=state_size, 
                    action_size=action_size,
                    add_noise=False) for _ in range(num_agents)]
    # Load trained parameters
    for i, agent in enumerate(agents):
        if i == 0:
            agent.actor_local.load_state_dict(torch.load(actor_1_params_file))
            agent.critic_local.load_state_dict(torch.load(critic_1_params_file))
        elif i == 1:
            agent.actor_local.load_state_dict(torch.load(actor_2_params_file))
            agent.critic_local.load_state_dict(torch.load(critic_2_params_file))
        else:
            print('Wrong number of agents - two are expected!')
            sys.exit(1)

    states = env_info.vector_observations              # Get the current states
    scores = np.zeros(num_agents)                      # Initialize the scores
    try:
        while True:
            actions = np.vstack(
                [agent.act(state.reshape(1, -1)) 
                for agent, state in zip(agents, states)]) # Select next actions                    
            env_info = env.step(actions)[brain_name]       # Send the actions to 
                                                           # environment
            next_states = env_info.vector_observations     # Get the next states
            rewards = env_info.rewards                     # Get the rewards
            scores += rewards                              # Update scores
            states = next_states                           # Roll over the states
                                                           # to next time step
    except UnityActionException:
        # Episode is completed
        pass

    # Print final scores
    print("Scores of each agent: {}".format(scores))
    print('Total score (averaged over agents): {}'.format(scores.mean()))
    # Close environment after three seconds
    time.sleep(3)
    env.close()


if __name__ == '__main__':
    # Process user arguments to start the requested function
    parser = argparse.ArgumentParser(
        description="Train or run an agent for/in the reacher environment")
    parser.add_argument('action', type=str,
                        help="Enter 'train' for agent training and 'run' for \
                              running a trained agent.")
    parser.add_argument('--actor_1_params', type=str,
                        default='checkpoint_actor_0.pth',
                        help="Filepath of the first actor-network-parameters \
                              to be used in the agent's model to run the game.")
    parser.add_argument('--critic_1_params', type=str,
                        default='checkpoint_critic_0.pth',
                        help="Filepath of the first critic-network-parameters \
                              to be used in the agent's model to run the game.")
    parser.add_argument('--actor_2_params', type=str,
                        default='checkpoint_actor_1.pth',
                        help="Filepath of the second actor-network-parameters \
                              to be used in the agent's model to run the game.")
    parser.add_argument('--critic_2_params', type=str,
                        default='checkpoint_critic_1.pth',
                        help="Filepath of the second critic-network-parameters \
                              to be used in the agent's model to run the game.")
    
    parser.add_argument('--watch', action='store_true',
                        help='Watch agent during training.')

    args = parser.parse_args()
    
    # Run the game with a smart agents
    if args.action == 'run':
        run_model(args.actor_1_params, args.critic_1_params,
                  args.actor_2_params, args.critic_2_params)
    # Train agents
    elif args.action == 'train':
        scores, avg_scores = train_agent(args.watch)
        # Plot scores and average
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(np.arange(1, len(scores)+1), 
                scores, c="blue", label="Score")
        ax.plot(np.arange(1, len(scores)+1), avg_scores, 
                c="red", label="Avg Score (last 100)")
        ax.set_xlabel("Episode #")
        ax.set_ylabel("Score")
        ax.legend()
        plt.show()
    else:
        print('Unkown action! Possible actions are "run" and "train".')

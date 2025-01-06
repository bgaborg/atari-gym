import datetime
import json
from pathlib import Path
import gymnasium as gym
import ale_py
import numpy as np
import wrappers
import metrics
import agent
import discord_bot

## CONFIGURATION
with open('config.json', 'r') as f:
    config = json.load(f)
CHECKPOINT_PATH = config['checkpoint_path']
DEBUG_SAVE_OBSERVATION = False
LIMIT_ACTION_SPACE = False
RENDER_MODE_HUMAN = False
TRAINING_EPISODES = 40000

## ENVIRONMENT
gym.register_envs(ale_py)
GAME_NAME = "Skiing-v5"
env = gym.make(f"ALE/{GAME_NAME}", render_mode="human" if RENDER_MODE_HUMAN else None)
env = wrappers.SkipFrame(env, skip=4)
env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)
env = gym.wrappers.ResizeObservation(env, shape=(84, 84))
env = gym.wrappers.FrameStackObservation(env, 4)
env = gym.wrappers.TransformObservation(env, lambda obs: obs / 255., observation_space=env.observation_space)

if LIMIT_ACTION_SPACE:
    allowed_actions = [2,3]
    env = wrappers.ActionSpaceWrapper(env, allowed_actions)
    for action in allowed_actions:
        print(f"Action {action}: {env.unwrapped.get_action_meanings()[action]}")
print(f"Action space: {env.action_space}")
observation, info = env.reset()

## CHECKPOINTS AND AGENT
Path(CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True)
checkpoints = sorted(Path(CHECKPOINT_PATH).iterdir(), key=lambda x: x.name, reverse=True)
# sort checkpoints by name
# in this directory the checkpoinst are saved in .chkpt file extension. find the last checkpoint
last_checkpoint = None
if len(checkpoints) > 0:
    for possible_checkpoint in checkpoints:
        if len(list(possible_checkpoint.glob('*.chkpt'))) > 0:
            last_checkpoint = sorted(possible_checkpoint.glob('*.chkpt'), key=lambda x: x.stat().st_ctime, reverse=True)
            if len(last_checkpoint) > 0:
                last_checkpoint = last_checkpoint[0]
                break
save_dir = Path(CHECKPOINT_PATH+f"_{GAME_NAME}") / datetime.datetime.now().strftime('%Y-%m-%d')
print(f"Last checkpoint: {last_checkpoint}")
save_dir.mkdir(parents=True, exist_ok=True)
logger = metrics.MetricLogger(save_dir=save_dir)

episode_times = np.zeros(TRAINING_EPISODES)
agent = agent.Agent(
    state_dim=(4, 84, 84),
    action_dim=int(env.action_space.n),
    save_dir=save_dir,
    iterations=TRAINING_EPISODES,
    checkpoint=last_checkpoint
)

if DEBUG_SAVE_OBSERVATION:
    # create the img directory if it doesn't exist
    Path("img").mkdir(parents=True, exist_ok=True)
    # delete all fles in the img directory
    for file in Path("img").iterdir():
        file.unlink()

total_start_time = datetime.datetime.now()
discord_bot.send(f"Training has started at {total_start_time}.")
for e in range(TRAINING_EPISODES):
    observation, info = env.reset()

    comm_reward = 0

    episode_start_time = datetime.datetime.now()
    while True:
        # 3. Show environment (the visual)
        if RENDER_MODE_HUMAN:
            env.render()

        # 4. Run agent on the state
        action = agent.act(observation)

        # 5. Agent performs action
        # receiving the next observation, reward and if the episode has terminated or truncated
        next_observation, reward, terminated, truncated, info = env.step(action)
        comm_reward += reward

        # 6. Remember
        agent.cache(state=observation, next_state=next_observation, action=action, reward=reward, done=(terminated or truncated))

        # 7. Learn
        q, loss = agent.learn()

        # 8. Logging
        logger.log_step(reward, loss, q)

        # 9. Update state
        observation = next_observation

        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated:
            break

        # save observation matrixes as text files to debug
        if DEBUG_SAVE_OBSERVATION:
            if e == 0:
                # remove the contents of the img directory
                env.unwrapped.ale.saveScreenPNG(f"img/{agent.curr_step}_ale_observation.png")
                for i in range(4):
                    # if any of the observation matrixes contain at least one non-zero value, save it as a text file
                    if np.any(observation[i]):
                        # save matrix as text file, with float64 precision
                        np.savetxt(f"img/{agent.curr_step}_observation_{i}.txt", observation[i], fmt='%.16f')

    logger.log_episode()

    delta_time = datetime.datetime.now() - episode_start_time
    episode_times[e] = delta_time.total_seconds()

    if e % 20 == 0:
        print(f"Episode {e} finished with reward {comm_reward} in {delta_time.total_seconds()} seconds.")
        average_episode_time = np.mean(episode_times[:e+1])
        expected_time_remaining = str(datetime.timedelta(seconds=(TRAINING_EPISODES - e) * average_episode_time))
        print(f"The average episode time is {average_episode_time} seconds, the expected time remaining is {expected_time_remaining}.")
        logger.record(
            episode=e,
            epsilon=agent.exploration_rate,
            step=agent.curr_step
        )

    if e % 1000 == 0:
        app_running_time = datetime.datetime.now() - total_start_time
        msg = (
            f"Training has been running for {app_running_time.total_seconds()/60} minutes. "
            f"Episode {e} finished with reward {comm_reward} in {delta_time.total_seconds()} seconds."
            f"The average episode time is {average_episode_time} seconds, the expected time remaining is {expected_time_remaining}."
            f"Epsilon: {agent.exploration_rate}, Step: {agent.curr_step}, Q: {q}, Loss: {loss}"
        )
        discord_bot.send(msg)

agent.save()
env.close()
discord_bot.send(f"Training has finished at {datetime.datetime.now()}, in {app_running_time.total_seconds()} seconds.")
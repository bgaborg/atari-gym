import datetime
from pathlib import Path
import gymnasium as gym
import ale_py
import numpy as np
import wrappers
import metrics
import agent
import discord_bot

gym.register_envs(ale_py)

env = gym.make('ALE/SpaceInvaders-v5')
env = wrappers.SkipFrame(env, skip=4)
env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)
env = gym.wrappers.ResizeObservation(env, shape=(84, 84))
env = gym.wrappers.TransformObservation(env, lambda obs: (obs / 255.0).astype(np.uint8), observation_space=env.observation_space)
env = gym.wrappers.FrameStackObservation(env, 4)
env = wrappers.ActionSpaceWrapper(env, [1,2,3,4,5])
observation, info = env.reset()

# # The environment's action space
# print(f"Action space: {env.action_space}")

# print("Possible actions:")
# for action in range(env.action_space.n):
#     print(f"Action {action}: {env.unwrapped.get_action_meanings()[action]}")

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)
logger = metrics.MetricLogger(save_dir=save_dir)

episodes = 40000
episode_times = np.zeros(episodes)

agent = agent.Agent(
    state_dim=(4, 84, 84),
    action_dim=env.action_space.n,
    save_dir=save_dir,
    iterations=episodes,
    checkpoint=None
)

total_start_time = datetime.datetime.now()
discord_bot.send(f"Training has started at {total_start_time}.")
for e in range(episodes):
    observation, info = env.reset()

    comm_reward = 0

    episode_start_time = datetime.datetime.now()
    while True:
        # 3. Show environment (the visual)
        # env.render()

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

    logger.log_episode()

    delta_time = datetime.datetime.now() - episode_start_time
    episode_times[e] = delta_time.total_seconds()

    if e % 20 == 0:
        print(f"Episode {e} finished with reward {comm_reward} in {delta_time.total_seconds()} seconds.")
        average_episode_time = np.mean(episode_times[:e+1])
        expected_time_remaining = str(datetime.timedelta(seconds=(episodes - e) * average_episode_time))
        print(f"The average episode time is {average_episode_time} seconds, the expected time remaining is {expected_time_remaining}.")
        logger.record(
            episode=e,
            epsilon=agent.exploration_rate,
            step=agent.curr_step
        )

    app_running_time = datetime.datetime.now() - total_start_time
    if app_running_time.total_seconds() > 30 * 60:
        # report current status to Discord
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
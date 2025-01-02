import gymnasium as gym
from gymnasium.spaces import Discrete

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
            if done:
                break
        return obs, reward, terminated, truncated, info

class ActionSpaceWrapper(gym.ActionWrapper):
    def __init__(self, env, allowed_actions):
        super(ActionSpaceWrapper, self).__init__(env)
        self.allowed_actions = allowed_actions
        self.action_space = Discrete(len(allowed_actions))

    def action(self, action):
        return self.allowed_actions[action]
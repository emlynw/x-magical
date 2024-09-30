import gym
from gym.wrappers import TimeLimit
from absl import app, flags
import os
from xmagical import register_envs
from xmagical.utils import KeyboardEnvInteractor
import cv2
import time
import pyglet.window

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "env_name",
    "SweepToTop-Gripper-State-Allo-TestLayout-v0",
    "The environment to load.",
)
flags.DEFINE_boolean("exit_on_done", True, "Whether to exit if done is True.")

image_path = '/home/emlyn/datasets/xmagical_custom_backwards/'
# cv2.imwrite(os.path.join(new_dir, str(i) + '.png'), cv2.cvtColor(obs, cv2.COLOR_BGR2RGB))
def main(_):
    num_eps = 3

    def step(action, dir):
            obs, rew, done, info = env.step(action)
            if obs.ndim != 3:
                obs = env.render("rgb_array")
                cv2.imwrite(os.path.join(dir, str(i[0]) + '.png'), cv2.cvtColor(obs, cv2.COLOR_BGR2RGB))
            if done and FLAGS.exit_on_done:
                return
            i[0] += 1
            return obs

    last_time = time.time()

    for j in range(num_eps):
        register_envs()
        env = gym.make(FLAGS.env_name)
        viewer = KeyboardEnvInteractor(action_dim=env.action_space.shape[0])
        env.reset()
        obs = env.render("rgb_array")
        viewer.imshow(obs)
        i = [0]

        new_dir = os.path.join(image_path, str(j+1))
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        while not viewer._finish_early:
            action = viewer.get_action()
            if viewer._started:
                obs = step(action, new_dir)
                if obs is None:
                    break
                viewer.imshow(obs)
            else:
                # Needed to run the event loop.
                viewer.imshow(viewer._last_image)
            pyglet.clock.tick()  # pytype: disable=module-attr
            delta = time.time() - last_time
            time.sleep(max(0, viewer._dt - delta))
            last_time = time.time()


if __name__ == "__main__":
    app.run(main)

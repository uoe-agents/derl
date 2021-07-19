import argparse
import time

import gym
import hallway_explore


def get_action():
    act = input("Action to select (in [0, 2]):")
    def check_valid(act):
        try:
            act = int(act)
        except:
            return False
        if act < 0 or act > 2:
            return False
        return True

    while not check_valid(act):
        act = input("Invalid action. Action to select (in [0, 2]):")

    return int(act)


def _game_loop(env, human, render):
    """
    Run one episode within environment
    """
    obs = env.reset()
    done = False

    if render:
        env.render()
        if not human:
            time.sleep(0.5)

    while not done:
        if human:
            action = get_action()
        else:
            action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        if reward > 0:
            print(f"Reward: {reward}")

        if render:
            env.render()
            if not human:
                time.sleep(0.5)


def main(game_count=1, human=False, render=False):
    env = gym.make("hallwayexp-10-10-40-v0")
    obs = env.reset()

    if human:
        render = True

    for episode in range(game_count):
        _game_loop(env, human, render)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the hallway-exploration environment.")

    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--times", type=int, default=1, help="How many times to run the game"
    )
    parser.add_argument("--human", action="store_true")

    args = parser.parse_args()
    main(args.times, args.human, args.render)
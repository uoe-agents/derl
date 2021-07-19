from gym.envs.registration import registry, register, make, spec
from itertools import product

goal_left = range(5, 51, 5)
goal_right = range(0, 51, 5)
ep_length = range(5, 101, 5)
random_start = [False, True]

for g_l, g_r, e_l, r_s in product(goal_left, goal_right, ep_length, random_start):
    if e_l < g_l:
        continue
    register(
        id="hallwayexp{3}-{0}-{1}-{2}-v0".format(g_l, g_r, e_l, "-random" if r_s else ""),
        entry_point="hallway_explore.hallwayexp:HallwayExplore",
        kwargs={
            "goal_left_length": g_l,
            "goal_right_length": g_r,
            "episode_length": e_l,
            "randomise_start_location": r_s,
        },
    )

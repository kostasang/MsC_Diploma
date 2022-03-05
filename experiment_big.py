import matplotlib.pyplot as plt, logging, joblib

from pyglet.window import key

from gym_duckietown.simulator import Simulator

from diploma_framework.algorithms import StackedFramePPO
from big_experiment_utils.utils import test_duckietown, test_duckietown_pedestrians
from big_experiment_utils.wrappers import *
from big_experiment_utils.model import CNNActorCritic

logger = logging.getLogger('gym-duckietown')
logger.setLevel(logging.WARNING)

env = Simulator(
    seed=None,  # random seed
    map_name="loop_pedestrians",
    max_steps=3_500,  
    domain_rand=False,
    distortion=False,
    camera_width=80,
    camera_height=60,
    draw_curve=True,
    accept_start_angle_deg=4
)
env = DtRewardWrapper(env)
env = DiscreteWrapper(env)

model = joblib.load('models/golden/direct/duckieTown_PPO_simple.joblib')
#model = CNNActorCritic(3, 'cpu')

alg = StackedFramePPO(environment=env,
          model=model,
          lr=2e-05,
          batch_size=8,
          epochs=3,
          max_frames=800_000,
          num_steps=500,
          clip_param=0.2,
          gamma=0.99,
          lamb= 1,
          actor_weight=1,
          critic_weight=0.5,
          entropy_weight=0.01,
          stacked_frames=5)

rewards_ppo, frames_ppo = alg.run(eval_window=5_000,
                                  n_evaluations=5,
                                  early_stopping=True,
                                  reward_threshold=35_000,
                                  frames_threshold=350_000,
                                  return_best=True,
                                  test_function=test_duckietown_pedestrians)

alg.save_model('models/duckieTown_PPO_simple.joblib')

plt.plot(list(range(5, (len(rewards_ppo)+1)*5, 5)), rewards_ppo)
plt.grid('minor')
plt.ylabel('Average reward')
plt.xlabel('Frames x1000')
plt.title('PPO Agent on DuckieTown')
plt.savefig('results/DuckieTown_PPO_rewards', dpi=500)

plt.figure()
plt.plot(list(range(5, (len(rewards_ppo)+1)*5, 5)), frames_ppo)
plt.grid('minor')
plt.ylabel('Average # of frames')
plt.xlabel('Frames x1000')
plt.title('PPO Agent on DuckieTown')
plt.savefig('results/DuckieTown_PPO_frames', dpi=500)

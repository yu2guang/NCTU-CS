#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from deep_rl import *


# DQN
def dqn_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.0005)
    config.network_fn = lambda: VanillaNet(config.action_dim, FCBody(config.state_dim))
    # config.network_fn = lambda: DuelingNet(config.action_dim, FCBody(config.state_dim))
    # config.replay_fn = lambda: Replay(memory_size=int(1e4), batch_size=10)
    config.replay_fn = lambda: AsyncReplay(memory_size=int(5e3), batch_size=128)

    config.random_action_prob = LinearSchedule(1.0, 0.1, 1e5)
    config.discount = 0.95
    config.target_network_update_freq = 50
    config.exploration_steps = 1000
    config.double_q = True
    config.double_q = False
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.eval_interval = int(5e3)
    config.max_steps = 1e5
    config.async_actor = False

    config.mini_batch_size = 128
    config.eval_episodes = 100
    run_steps(DQNAgent(config))


# # N-Step DQN
# def n_step_dqn_feature(**kwargs):
#     generate_tag(kwargs)
#     kwargs.setdefault('log_level', 0)
#     config = Config()
#     config.merge(kwargs)

#     config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
#     config.eval_env = Task(config.game)
#     # config.num_workers = 5
#     config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.0005)
#     config.network_fn = lambda: VanillaNet(config.action_dim, FCBody(config.state_dim))
#     config.replay_fn = lambda: AsyncReplay(memory_size=int(5e3), batch_size=128)  ###
#     config.random_action_prob = LinearSchedule(1.0, 0.1, 1e5)
#     config.discount = 0.95
#     config.target_network_update_freq = 50
#     config.exploration_steps = 1000  ###
#     config.rollout_length = 3
#     config.gradient_clip = 5
#     # config.eval_interval = int(5e3)  ###
#     config.async_actor = False  ###
    
#     config.max_steps = 1e5
#     config.mini_batch_size = 128
#     run_steps(NStepDQNAgent(config))


# DDPG
def ddpg_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.max_steps = int(150000)
    config.eval_interval = int(1e4)
    config.eval_episodes = 20

    config.network_fn = lambda: DeterministicActorCriticNet(
        config.state_dim, config.action_dim,
        actor_body=FCBody(config.state_dim, (400, 300), gate=F.relu),
        critic_body=TwoLayerFCBodyWithAction(
            config.state_dim, config.action_dim, (400, 300), gate=F.relu),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3))

    config.replay_fn = lambda: Replay(memory_size=int(1e4), batch_size=64)
    config.discount = 0.99
    config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(
        size=(config.action_dim,), std=LinearSchedule(0.2))
    config.warm_up = int(1e4)
    config.target_network_mix = 1e-3    # tau

    config.mini_batch_size = 64
    run_steps(DDPGAgent(config))


if __name__ == '__main__':
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    random_seed()
    # select_device(-1)   # cpu
    select_device(0)

    ### DQN
    game = 'CartPole-v0'
    print('DQN: %s'%game)
    dqn_feature(game=game)
    # print('N-step DQN: %s'%game)
    # n_step_dqn_feature(game=game)

    ### DDPG
    game = 'Pendulum-v0'
    print('DDPG: %s'%game)
    ddpg_continuous(game=game)

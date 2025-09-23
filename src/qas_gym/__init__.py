from gymnasium.envs.registration import register

# 注册 BasicTwoQubitEnv
register(
    id='BasicTwoQubit-v0',
    entry_point='qas_gym.envs.basic_envs:BasicTwoQubitEnv',
)

# 注册 BasicThreeQubitEnv
register(
    id='BasicThreeQubit-v0',
    entry_point='qas_gym.envs.basic_envs:BasicThreeQubitEnv',
)

# 注册 BasicNQubitEnv (更通用，可能需要用户指定 target)
register(
    id='BasicNQubit-v0',
    entry_point='qas_gym.envs.basic_envs:BasicNQubitEnv',
    kwargs={
        'target': None,  # 需要在使用时指定
    }
)


# 注册 NoisyTwoQubitEnv
register(
    id='NoisyTwoQubit-v0',
    entry_point='qas_gym.envs.noisy_envs:NoisyTwoQubitEnv',
)


# 注册 NoisyThreeQubitEnv
register(
    id='NoisyThreeQubit-v0',
    entry_point='qas_gym.envs.noisy_envs:NoisyThreeQubitEnv',
)


# 注册 NoisyNQubitEnv (更通用，需要指定 target, error_gates, error_observables)
register(
    id='NoisyNQubit-v0',
    entry_point='qas_gym.envs.noisy_envs:NoisyNQubitEnv',
    kwargs={
        'target': None,  # 需要在使用时指定
        'error_gates': None, # 需要在使用时指定
        'error_observables': None # 需要在使用时指定
    }
)


# 注册 QuantumArchSearchEnv (更通用，需要指定所有参数)
register(
    id='QuantumArchSearch-v0',
    entry_point='qas_gym.envs.qas_env:QuantumArchSearchEnv',
        kwargs={
        'target': None,
        'qubits': None,
        'state_observables': None,
        'action_gates': None,
        'fidelity_threshold': 0.9,  # 替换为你的默认值或在创建环境时指定
        'reward_penalty': 0.01,  # 替换为你的默认值或在创建环境时指定
        'max_timesteps': 100,
        'error_observables': None,
        'error_gates': None,
        'seed': None,
    },

)
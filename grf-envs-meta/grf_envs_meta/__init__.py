from gym.envs.registration import register

register(
    id='GRFWrapper-v0',
    entry_point='grf_envs_meta.grf_wrapper_env:GRFWrapperEnv',
)


from gym.envs.registration import register

register(
    id='PP_Multi_Task-v0',
    entry_point='pp_multi_envs.pp_wrapper_env:PPWrapperEnv',
)



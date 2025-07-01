from easydict import EasyDict

m4o_hybrid_hppo_config = dict(
    # the path to store the training result
    exp_name='hppo_trans/1',
    env=dict(
        collector_env_num=5,
        evaluator_env_num=1,
        act_scale=True,
        env_id='M4O-v0',
        # n_evaluator_episode=5,
        # stop_value=90,
    ),
    policy=dict(
        cuda=True,
        action_space='hybrid',
        # recompute_adv=True,
        model=dict(
            obs_shape=47, # shape of state
            action_shape=dict(
                action_type_shape=512,
                action_args_shape=3,
            ),
            action_space='hybrid',
            encoder_hidden_size_list=[256,128,64,64], # hidden layer
            # sigma_type='fixed',
            # fixed_sigma_value=0.3,
            bound_type='tanh',
        ),
        learn=dict(
            # epoch_per_collect=10,
            batch_size=64,
            learning_rate=5e-4,
            # entropy_weight=0.5,
            # adv_norm=True,
            # value_norm=True,
        ),
        collect=dict(
            n_sample=512,
            # discount_factor=0.2,
            # gae_lambda=0.95,
            collector=dict(collect_print_freq=100, ),
        ),
        # eval=dict(evaluator=dict(eval_freq=200, ), ),
    ),
)
m4o_hybrid_hppo_config = EasyDict(m4o_hybrid_hppo_config)
main_config = m4o_hybrid_hppo_config

m4o_hybrid_hppo_create_config = dict(
    env=dict(
        type='m4o_hybrid',
        import_names=['dizoo.m4o_hybrid.envs.m4o_hybrid_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo'),
)
m4o_hybrid_hppo_create_config = EasyDict(m4o_hybrid_hppo_create_config)
create_config = m4o_hybrid_hppo_create_config

if __name__ == "__main__":
    # tensorboard --logdir=./log
    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy([main_config, create_config], seed=0, max_env_step=int(3e5))

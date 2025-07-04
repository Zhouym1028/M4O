from typing import Union, Optional, List, Any, Tuple
import os
import torch
from ditk import logging
from functools import partial
from tensorboardX import SummaryWriter
from copy import deepcopy

from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner, InteractionSerialEvaluator, BaseSerialCommander, create_buffer, \
    create_serial_collector
from ding.config import read_config, compile_config
from ding.policy import create_policy, PolicyFactory
from ding.reward_model import create_reward_model
from ding.utils import set_pkg_seed


def serial_pipeline_onpolicy(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':  # noqa
    """
    Overview:
        Serial pipeline entry on-policy RL.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - env_setting (:obj:`Optional[List[Any]]`): A list with 3 elements: \
            ``BaseEnv`` subclass, collector env config, and evaluator env config.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - max_train_iter (:obj:`Optional[int]`): Maximum policy update iterations in training.
        - max_env_step (:obj:`Optional[int]`): Maximum collected environment interaction steps.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """
    # cfg = main_config  create_config = create_cfg
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = deepcopy(input_cfg)

    #ppo_command
    create_cfg.policy.type = create_cfg.policy.type + '_command'

    # env_fn = None
    env_fn = None if env_setting is None else env_setting[0]

    # 生成最终的 cfg
    cfg = compile_config(
        cfg,
        seed=seed,
        env=env_fn,
        auto=True,
        create_cfg=create_cfg,
        save_cfg=True,
        renew_dir=not cfg.policy.learn.get('resume_training', False)
    )

    # Create main components: env, policy
    # env_fn 环境生成函数
    # collector_env_cfg / evaluator_env_cfg：用于采集和评估的环境参数
    if env_setting is None:
        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    else:
        env_fn, collector_env_cfg, evaluator_env_cfg = env_setting

    # collector_env 负责训练数据的采集
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    # evaluator_env 用于周期性评估策略性能
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)

    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    # 创建策略网络
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval', 'command'])

    # Create worker components: learner, collector, evaluator, replay buffer, commander.
    # 创建tensorboard 记录数据
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))

    # learner 负责从收集器中接收数据，并对策略（Policy）进行优化 策略更新
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    # collector 用于数据收集，为策略训练提供样本数据  数据采样
    collector = create_serial_collector(
        cfg.policy.collect.collector,
        env=collector_env,
        policy=policy.collect_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name
    )
    # evaluator评估当前策略的性能，判断策略是否达到目标表现
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    # commander调控训练流程的核心模块，确保各个组件按顺序协同工作
    commander = BaseSerialCommander(
        cfg.policy.other.commander, learner, collector, evaluator, None, policy.command_mode
    )

    # ==========
    # Main loop
    # ==========
    # Learner's before_run hook.
    learner.call_hook('before_run')
    if cfg.policy.learn.get('resume_training', False):
        collector.envstep = learner.collector_envstep

    while True:
        collect_kwargs = commander.step()
        # Evaluate policy performance
        # 评估当前策略性能 并在一定的iter后保存chkpt
        if evaluator.should_eval(learner.train_iter):
            stop, eval_info = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break

        # Collect data by default config n_sample/n_episode 数据采集
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)

        # Learn policy from collected data 策略学习更新 优化 Actor 和 Critic 网络权重
        learner.train(new_data, collector.envstep)
        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break

    # Learner's after_run hook.
    learner.call_hook('after_run')
    import time
    import pickle
    import numpy as np
    with open(os.path.join(cfg.exp_name, 'result.pkl'), 'wb') as f:
        eval_value_raw = eval_info['eval_episode_return']
        final_data = {
            'stop': stop,
            'env_step': collector.envstep,
            'train_iter': learner.train_iter,
            'eval_value': np.mean(eval_value_raw),
            'eval_value_raw': eval_value_raw,
            'finish_time': time.ctime(),
        }
        pickle.dump(final_data, f)
    return policy

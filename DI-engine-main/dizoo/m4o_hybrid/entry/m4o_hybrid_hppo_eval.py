import os
import torch
from tensorboardX import SummaryWriter
from easydict import EasyDict
from functools import partial

from ding.config import compile_config
from ding.worker import InteractionSerialEvaluator
from ding.envs import BaseEnvManager
from ding.envs import get_vec_env_setting
from ding.policy import PPOPolicy
from ding.model import VAC
from ding.utils import set_pkg_seed
from dizoo.m4o_hybrid.config.m4o_hybrid_hppo_config import m4o_hybrid_hppo_config, m4o_hybrid_hppo_create_config

def main(main_cfg, create_cfg, seed=0):
    # 指定测试时使用的模型权重和保存回放的路径
    main_cfg.policy.load_path = '../config/m4o_hybrid_hppo_seed0/ckpt/ckpt_best.pth.tar'  # 已训练好的模型路径
    main_cfg.env.evaluator_env_num = 1  # 测试环境数量
    cfg = compile_config(main_cfg, seed=seed, auto=True, create_cfg=create_cfg, save_cfg=True)

    # 创建评估环境
    env_fn, _, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    evaluator_env = BaseEnvManager([partial(env_fn, cfg=c) for c in evaluator_env_cfg], cfg.env.manager)


    # 设置随机种子
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    # 构建策略并加载模型
    model = VAC(**cfg.policy.model)
    policy = PPOPolicy(cfg.policy, model=model)
    policy.eval_mode.load_state_dict(torch.load(cfg.policy.load_path, map_location='cpu'))  # 加载权重

    # 创建评估器
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )

    # 开始评估
    evaluator.eval()

if __name__ == "__main__":
    # 渲染前请确保 gym 版本兼容（低于 0.22.0 以避免渲染 API 弃用问题）
    main(m4o_hybrid_hppo_config, m4o_hybrid_hppo_create_config, seed=0)

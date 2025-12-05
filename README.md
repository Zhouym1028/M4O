# M4O

The A2C used comes from Stable Baselines 3, and the training environment is Gym. 

The packages required to run the A2C algorithm in the experiments are already listed in the `readme.md` file under the `M4O_indivisible` directory.

The M4O-PO-related code is located in `DI-engine-main\dizoo\m4o_hybrid`. 
To run the code, you need to install the required packages first by [pip install requests -i https://mirrors.aliyun.com/pypi/simple/ DI-engine]. 
The files in `m4o_hybrid\envs\m4o-hybrid\m4o_hybrid` constitute the training environment, while `m4o_hybrid\config\m4o_hybrid_hppo_config.py` is used to configure the hyperparameters of the H-PPO algorithm and also serves as the training launch script. Additionally, `m4o_hybrid\envs\m4o_hybrid_env.py` adapts the H-PPO algorithm to the corresponding action and state dimensions.

The detailed usage of the H-PPO algorithm can be found at: [https://github.com/opendilab/DI-engine].

"""MLP trained for 10,000 optimizer steps; also save the 5,000-step snapshot."""

CONFIG = {'data_root': 'data/tartan-imu-challenge-iros2026',
 'output_dir': 'runs/mlp_10k_drone',
 'seed': 42,
 'n_splits': 5,
 'folds': [0, 1, 2, 3, 4],
 'platform': 'drone',
 'hidden_sizes': [128, 64],
 'batch_size': 256,
 'learning_rate': 0.001,
 'weight_decay': 0.0001,
 'device': 'cpu',
 'num_threads': 4,
 'eval_every_steps': 250,
 'drop_last': True,
 'max_steps': 10000,
 'snapshot_steps': [5000, 10000]}

 Workflow to run:
  # 1. Generate PIT episodes
  python src/generate_pit_dataset.py dataset.n_episodes=5000
  # 2. Train
  python src/train.py training.dataset_dir=./data/pit_episodes
  # 3. Evaluate on UCI
  python src/evaluate.py --ckpt ./checkpoints/copula_transformer/step_0029999_final.pt


Use the conda environment 'multivariate_ICL' to run some test locally. 
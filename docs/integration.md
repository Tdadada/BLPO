# Trainer integration

BLPO replaces the advantage-estimation stage of an on-policy agent trainer. Rollout generation, reward computation, log-probability recomputation, PPO/GRPO actor updates, and distributed execution remain the responsibility of the host runtime.

## Required order within an update

1. Generate eight trajectories for each of the 16 sampled tasks.
2. Flatten environment decisions into one batch row per action.
3. Compute scalar environment rewards and discounted decision returns.
4. Call `BLPOEstimator.compute` once. The estimator reads the persistent memory, incorporates the current batch, and returns token-broadcast advantages.
5. Run the actor update with the returned `advantages` tensor.
6. Save `blpo_memory.json` whenever an actor checkpoint is saved.

The reported implementation incorporates the current batch before querying task-progress statistics. WebShop additionally centers residuals within the current batch, as specified in its configuration.

## Batch interface

The object passed as `batch` must implement `__len__` and expose:

- `batch["input_ids"]`: tensor used to select the output device;
- `non_tensor_batch["uid"]`: sampled-task group identifier;
- `non_tensor_batch["traj_uid"]`: trajectory identifier;
- `non_tensor_batch["anchor_obs"]`: observation before the current action;
- `non_tensor_batch["text_action"]`: executed action text;
- `non_tensor_batch["rewards"]`: scalar environment reward per decision;
- `non_tensor_batch["is_format_valid"]`: whether the action obeys the required format;
- `non_tensor_batch["is_action_admissible"]`: whether the environment can execute the action.

ALFWorld also requires `gamefile`. WebShop instead requires `webshop_task` and `webshop_pre_public_state`. The WebShop public state is the visible page state before the action; it must not contain privileged future information.

For optional Parquet audit records, also provide `episode_rewards` and, for WebShop, `webshop_won`.

## Checkpoint contract

`BLPOEstimator.save(checkpoint_dir)` writes `blpo_memory.json` atomically. The file must travel with a resumable actor checkpoint. `BLPOEstimator.load(checkpoint_dir)` validates the memory format and EMA half-life before restoring it.

Inference-only checkpoints do not need the BLPO memory because the estimator is used only during training.

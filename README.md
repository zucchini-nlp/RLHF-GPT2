**Alignment Task with trl Library**

## Overview

This repository implements a Command Line Interface (CLI) for fine-tuning the [gpt2-medium](https://huggingface.co/gpt2-medium) model using the [trl](https://huggingface.co/docs/trl/index) library. The fine-tuning process supports two training options:

- Reinforcement Learning with Hybrid Fine-Tuning (RLHF) using Proximal Policy Optimization (PPO) with a reward model.
- RLHF using Deterministic Policy Optimization (DPO) without a reward model.

The implementation allows for a single iteration of the following stages:
1. SFT (Supervised Fine-Tuning)
2. (Optional) Reward Model Training
3. RL with PPO or DPO

## Usage

### Training Scripts

You can use the provided scripts to train different parts of the alignment iteration. For instance, to train the SFT model with a gpt2-medium backbone on the Anthropic-hh dataset, run the following command:

```bash
bash run_sft.sh
```

Similarly, you can execute other scripts such as `run_reward.sh` and `run_rlhf.sh` to train the reward model and perform RLHF, respectively.


### Experiment

To conduct a normal training on the [Anthropic-hh dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf), run the appropriate script or follow the provided examples.

Feel free to explore and modify the scripts/config to adapt the fine-tuning process according to your specific requirements.

### Generation
To generate using the fine-tuned models, use the below command:

```bash
python generate.py --pretrained_model_name_or_path RaushanTurganbay/GPT2_sft_and_dpo_tuned max_new_tokens 256 {{other generaion_args}}
```

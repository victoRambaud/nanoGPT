# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

out_dir = "out"
wandb_run_name = ""
ckpt_path = ""
ckpt_name = ""
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'

# wandb logging
wandb_log = True  # disabled by default
wandb_project = "gpt2-openweb"
wandb_mode = "offline"

# data
dataset = "openwebtext_data_folder"
gradient_accumulation_steps = 5 * 8  # used to simulate larger batch sizes
batch_size = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024

# model
n_layer = 12
head_dim = 64
n_embd = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
rope: bool = True
cope: bool = False
working_memory: bool = False
episodic_memory: bool = False
n_episodic_memory: bool = False
transformer_type: str = "nWM"
n_working_memory: bool = True
inv_scale_attn: bool = False
em_wm: bool = False
n_approx_steps: int = -1
dt_rank: int = 8
base_freq = block_size
block_max_init: float = 1.0
block_layer_scaling_ratio: float = 0.

# run name
if init_from == "scratch":
    if transformer_type == "WM":
        wandb_run_name = f"WM_L{n_layer}_n{n_embd}_base{base_freq}_rank{dt_rank}_ls{block_layer_scaling_ratio}"
    elif transformer_type == "EM":
        wandb_run_name = f"EM_L{n_layer}_n{n_embd}_base{base_freq}_rank{dt_rank}_ls{block_layer_scaling_ratio}"
    elif transformer_type == "nEMWM":
        wandb_run_name = f"nEMWM_invscale_{inv_scale_attn}_L{n_layer}_n{n_embd}_base{base_freq}_rank{dt_rank}_ls{block_layer_scaling_ratio}"
    elif transformer_type == "nWM" and not rope:
        wandb_run_name = f"nWM_invscale_{inv_scale_attn}_L{n_layer}_n{n_embd}_base{base_freq}_rank{dt_rank}_ls{block_layer_scaling_ratio}"
    elif transformer_type == "nWM" and rope:
        wandb_run_name = f"nROPE_invscale_{inv_scale_attn}_L{n_layer}_n{n_embd}_base{base_freq}_rank{dt_rank}_ls{block_layer_scaling_ratio}"
    elif transformer_type == "nEM":
        wandb_run_name = f"nEM_invscale_{inv_scale_attn}_L{n_layer}_n{n_embd}_base{base_freq}_rank{dt_rank}_ls{block_layer_scaling_ratio}"
    elif cope:
        wandb_run_name = f"COPE_L{n_layer}_n{n_embd}"
        transformer_type = "WM"
    else:
        rope = True
        wandb_run_name = f"ROPE_L{n_layer}_n{n_embd}"
        transformer_type = "WM"

# adamw optimizer
learning_rate = 3e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 4000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 1e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
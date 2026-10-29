out_dir = "out"
wandb_run_name = ""
ckpt_name = ""
ckpt_path = ""
eval_interval = 500
log_interval = 50
eval_iters = 50
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
eval_blimp = False

# wandb logging
wandb_log = True  # disabled by default
wandb_project = "babylm"
wandb_mode = "offline"

# data
dataset = "babylm"
gradient_accumulation_steps = 4  # used to simulate larger batch sizes
batch_size = 16  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 256

# model
n_layer = 4
head_dim = 64
n_embd = 256
dropout = 0.1  # small-data regularization for BabyLM
bias = False  # do we use bias inside LayerNorm and Linear layers?
rope: bool = False
cope: bool = False
fox: bool = True
working_memory: bool = True
episodic_memory: bool = False
n_episodic_memory: bool = False
n_working_memory: bool = False
follow_rank: bool = True
shared_inner_theta: bool = False
transformer_type = "WM"
dt_rank: int = 2
base_freq = 256
block_max_init: float = 1.0

# egoformer
commute = True
rotate_values = True
ego2allo = True
value_base_freq = 10
single_freq = True
gate_algebra = True
freq_grad = False
attention_type = "normal"
reg_loss_weight = 0.05

# run name
if init_from == "scratch":
    if transformer_type == "WM" and not rope:
        if ego2allo:
            prefix = "EGO"
        else:
            prefix = ""
        prefix += "_WM" if attention_type == "normal" else "_EM"
        wandb_run_name = f"{prefix}_L{n_layer}_n{n_embd}_base{base_freq}_rank{dt_rank}"
    # elif transformer_type == "EM":
    #     wandb_run_name = f"EM_L{n_layer}_n{n_embd}_base{base_freq}_rank{dt_rank}_ls{block_layer_scaling_ratio}"
    # elif transformer_type == "nEMWM":
    #     wandb_run_name = f"nEMWM_invscale_{inv_scale_attn}_L{n_layer}_n{n_embd}_base{base_freq}_rank{dt_rank}_ls{block_layer_scaling_ratio}"
    # elif transformer_type == "nWM" and not rope:
    #     wandb_run_name = f"nWM_invscale_{inv_scale_attn}_L{n_layer}_n{n_embd}_base{base_freq}_rank{dt_rank}_ls{block_layer_scaling_ratio}"
    # elif transformer_type == "nWM" and rope:
    #     wandb_run_name = f"nROPE_invscale_{inv_scale_attn}_L{n_layer}_n{n_embd}_base{base_freq}_rank{dt_rank}_ls{block_layer_scaling_ratio}"
    # elif transformer_type == "nEM":
    #     wandb_run_name = f"nEM_invscale_{inv_scale_attn}_qkpos{em_qk_positions}_L{n_layer}_n{n_embd}_base{base_freq}_rank{dt_rank}_ls{block_layer_scaling_ratio}"
    elif cope:
        wandb_run_name = f"COPE_L{n_layer}_n{n_embd}"
        transformer_type = "WM"
    else:
        rope = True
        wandb_run_name = f"ROPE_L{n_layer}_n{n_embd}"
        transformer_type = "WM"

# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 3000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 100  # how many steps to warm up for
lr_decay_iters = 3000  # should be ~= max_iters
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10
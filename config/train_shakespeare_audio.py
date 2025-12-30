# config/train_shakespeare_audio.py

out_dir = 'out-shakespeare-audio'
eval_interval = 250
eval_iters = 20
log_interval = 10
always_save_checkpoint = True

dataset = 'shakespeare_audio'

# --- THE MEMORY FIX ---
# We lower batch_size from 64 to 8 (8x less RAM usage)
# We increase gradient_accumulation_steps from 1 to 8 
# Effective batch size remains 64 (8 * 8 = 64), so training is stable.
gradient_accumulation_steps = 8 
batch_size = 8 
# ----------------------

block_size = 1024 

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4

device = 'mps'       
compile = False      
dtype = 'float32'    

vocab_size = 4096
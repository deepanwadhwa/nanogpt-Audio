import os
import torch
import torch.nn.functional as F
from model import GPTConfig, GPT
from encodec import EncodecModel
from scipy.io.wavfile import write

# --- CONFIG ---
out_dir = 'out-shakespeare-audio'
max_new_tokens = 3000   
temperature = 0.75         
top_k = 50                
device = 'cpu'    
seed = 1187
# --------------

torch.manual_seed(seed)

print(f"Loading model from {out_dir}...")
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)

state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.eval()
model.to(device)

print("Loading EnCodec...")
audio_model = EncodecModel.encodec_model_24khz()
audio_model.set_target_bandwidth(3.0)
audio_model.to(device)

print(f"Generating {max_new_tokens} tokens...")

# Feed [L0, L1, L2, L3] to force the model's prediction to align with the next frame start
start_frame = torch.tensor([[0, 1024, 2048, 3072]], dtype=torch.long, device=device)
idx = start_frame 

with torch.no_grad():
    for i in range(max_new_tokens):
        # Crop context
        idx_cond = idx if idx.size(1) <= gptconf.block_size else idx[:, -gptconf.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        
        if i % 50 == 0:
            print(f"Generated {i}/{max_new_tokens}...", end='\r')

print(f"\nDecoding...")

# Skip the first 4 tokens (the prompt we manually added)
out_tokens = idx[0, 4:].cpu()

# Un-flatten
n_layers = 4
valid_len = (len(out_tokens) // n_layers) * n_layers
out_tokens = out_tokens[:valid_len]
codes = out_tokens.view(-1, n_layers).t()

# Remove offsets
offset_vals = torch.tensor([0, 1024, 2048, 3072]).view(4, 1)
clean_codes = codes - offset_vals

# Clamp to be safe (0-1023 is the valid range)
clean_codes = torch.clamp(clean_codes, 0, 1023)

# Decode
final_codes = clean_codes.unsqueeze(0)
decoded_wav = audio_model.decode([(final_codes.to(device), None)])

# Detach memory to avoid error
audio_data = decoded_wav[0, 0].detach().cpu().numpy()

filename = 'aligned_sample.wav'
write(filename, 24000, audio_data)
print(f"Saved to {filename}")
import torch
import soundfile as sf
from encodec import EncodecModel
from encodec.utils import convert_audio
import numpy as np
import os
import gc

# --- Configuration ---
INPUT_WAV = "shakespeare.wav"
OUTPUT_BIN = "train.bin"
# Force CPU to avoid "NotImplementedError" on Mac MPS for specific convolutions
DEVICE = "cpu" 
CHUNK_DURATION = 60 # Process 60 seconds at a time to save RAM

def main():
    print(f"Loading EnCodec model on {DEVICE}...")
    model = EncodecModel.encodec_model_24khz()
    model.to(DEVICE)
    model.set_target_bandwidth(3.0) 
    model.eval()

    if not os.path.exists(INPUT_WAV):
        print(f"Error: {INPUT_WAV} not found.")
        return

    # Get file info without loading it
    info = sf.info(INPUT_WAV)
    sr = info.samplerate
    total_frames = info.frames
    duration = total_frames / sr
    
    print(f"Processing {INPUT_WAV}")
    print(f"Duration: {duration/3600:.2f} hours")
    print("-" * 30)

    # List to store the tiny resulting tokens
    all_codes = [] 

    # --- STREAMING LOOP ---
    # We use soundfile to read blocks instead of the whole file
    block_size = int(sr * CHUNK_DURATION)
    
    # Create a generator that reads the file in chunks
    with sf.SoundFile(INPUT_WAV) as f:
        chunk_idx = 0
        while f.tell() < total_frames:
            # Read a chunk
            wav_np = f.read(block_size)
            
            # Convert to Tensor
            wav = torch.from_numpy(wav_np).float().to(DEVICE)
            
            # Fix dimensions [Time] -> [1, 1, Time]
            if wav.ndim == 1:
                wav = wav.unsqueeze(0).unsqueeze(0)
            elif wav.ndim == 2:
                wav = wav.t().unsqueeze(0) # [1, Channels, Time]

            # Resample just this chunk
            wav = convert_audio(wav, sr, model.sample_rate, model.channels)

            # Tokenize
            with torch.no_grad():
                # encode returns list of (codes, scale)
                encoded_frames = model.encode(wav)
                
            # Extract codes: [1, n_codebooks, Time] -> [n_codebooks, Time]
            # We assume 1 frame per chunk usually, but we concat if multiple
            chunk_codes = torch.cat([frame[0] for frame in encoded_frames], dim=-1)
            chunk_codes = chunk_codes.squeeze(0).cpu() # Move to CPU storage
            
            all_codes.append(chunk_codes)
            
            # Progress bar logic
            chunk_idx += 1
            processed_sec = min((chunk_idx * CHUNK_DURATION), duration)
            print(f"Processed: {processed_sec:.1f}s / {duration:.1f}s ({(processed_sec/duration)*100:.1f}%)")

            # Force memory cleanup
            del wav
            gc.collect()

    print("Stitching tokens together...")
    # Concatenate all chunks along the Time dimension (dim=1)
    full_codes = torch.cat(all_codes, dim=1).to(DEVICE)
    
    # --- FLATTENING (The Nano Trick) ---
    n_codebooks = full_codes.shape[0]
    print(f"Final shape: {full_codes.shape}")
    
    codebook_size = 1024
    offsets = torch.tensor([i * codebook_size for i in range(n_codebooks)]).to(DEVICE)
    offsets = offsets.view(-1, 1)

    offset_codes = full_codes + offsets
    train_data = offset_codes.permute(1, 0).reshape(-1).cpu().numpy().astype(np.uint16)

    print(f"Generated {len(train_data)} tokens.")
    train_data.tofile(OUTPUT_BIN)
    print(f"Saved to {OUTPUT_BIN}")

if __name__ == "__main__":
    main()
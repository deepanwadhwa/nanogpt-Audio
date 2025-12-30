import asyncio
import os
import shutil
from edge_tts import Communicate
from pydub import AudioSegment

# Configuration
INPUT_FILE = "input.txt"
OUTPUT_FILE = "shakespeare.wav"
VOICE = "en-GB-SoniaNeural"
CHUNK_SIZE = 1000
TEMP_DIR = "temp_audio_chunks"

async def generate_chunk(text, index):
    """Generates audio for a single chunk of text."""
    filename = os.path.join(TEMP_DIR, f"chunk_{index:04d}.mp3")
    communicate = Communicate(text, VOICE)
    await communicate.save(filename)
    return filename

async def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found!")
        return
    
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

    # Read and clean text
    print("Reading text file...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        full_text = f.read()
    
    # Remove excessive newlines to make flow better
    clean_text = full_text.replace("\n", " ").replace("  ", " ")
    
    # Split into chunks
    chunks = [clean_text[i:i+CHUNK_SIZE] for i in range(0, len(clean_text), CHUNK_SIZE)]
    print(f"Split text into {len(chunks)} chunks.")

    # Generate Audio (Async)
    print("Generating audio segments... (this may take a while)")
    tasks = []
    for i, chunk in enumerate(chunks):
        tasks.append(generate_chunk(chunk, i))
    
    # Limit concurrent connections
    semaphore = asyncio.Semaphore(5) 
    
    async def limited_generate(text, idx):
        async with semaphore:
            print(f"Processing chunk {idx+1}/{len(chunks)}...")
            return await generate_chunk(text, idx)

    chunk_files = await asyncio.gather(*[limited_generate(c, i) for i, c in enumerate(chunks)])

    # Stitch audio using Pydub
    print("Stitching audio files together...")
    combined_audio = AudioSegment.empty()
    
    # Sort files to ensure correct order
    chunk_files.sort()
    
    for mp3_file in chunk_files:
        try:
            segment = AudioSegment.from_mp3(mp3_file)
            combined_audio += segment
        except Exception as e:
            print(f"Warning: Could not add {mp3_file}: {e}")

    # Export Final WAV (24kHz mono)
    print(f"Exporting to {OUTPUT_FILE}...")
    combined_audio = combined_audio.set_frame_rate(24000).set_channels(1)
    combined_audio.export(OUTPUT_FILE, format="wav")

    # Cleanup
    shutil.rmtree(TEMP_DIR)
    print("Done! You now have a talking Shakespeare file.")

if __name__ == "__main__":
    asyncio.run(main())
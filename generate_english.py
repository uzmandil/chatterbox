import torch
import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS
import os

# Detect device (Mac M1/M2/M3/M4 uses 'mps', otherwise 'cpu')
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load the Turbo model (English only, fast)
print("Loading model... (this might take a few minutes for the first time)")
model = ChatterboxTurboTTS.from_pretrained(device=device)

# The text you want to speak
text = "Hello! I am Chatterbox Turbo. I am now running locally on your Mac. How can I help you today?"

# Generate audio
# Note: Turbo can generate from text alone using a default reference, 
# or you can provide your own .wav file to clone a specific voice.
print("Generating audio...")
wav = model.generate(text)

# Save the output
output_file = "english_output.wav"
ta.save(output_file, wav, model.sr)

print(f"Success! Audio saved as: {os.path.abspath(output_file)}")

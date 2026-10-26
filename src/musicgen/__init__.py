import torch
from scipy.io.wavfile import write
from transformers import AutoProcessor, MusicgenForConditionalGeneration

MODEL_ID = "facebook/musicgen-small"
DEVICE = "cpu"

PROMPT = "Create a smooth, sophisticated jazz instrumental piece with a warm and relaxing atmosphere. Blend expressive saxophone melodies, mellow piano chords, soft double bass, gentle brushed drums, and subtle trumpet accents. The music should evoke the feeling of a cozy jazz club late at night, with dim lighting, rain falling outside, and a peaceful, elegant ambiance. Incorporate soulful improvisations, rich harmonies, a steady swing rhythm, and a natural, intimate sound. Keep the tempo moderate and the mood smooth, nostalgic, and slightly romantic, perfect for relaxing, studying, working, or enjoying a quiet evening. Use high-quality studio recording, warm analog textures, and balanced instrumentation. No vocals, no lyrics, instrumental jazz only."

print("Cargando Musicgen...")

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID)

model = model.to(DEVICE)
model.eval()

print("Preparando prompt...")

inputs = processor(text=[PROMPT], padding=True, return_tensors = "pt").to(DEVICE)

print("Generando musica...")

with torch.inference_mode():
    audio = model.generate(
        **inputs,
        do_sample=True,
        guidance_scale=3.0,
        max_new_tokens=1500
    )
sample_rate = model.config.audio_encoder.sampling_rate
waveform = audio[0,0].cpu().float().numpy()

write("musica.wav", sample_rate, waveform)
print("Musica generada")
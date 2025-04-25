import os
import json
import torch
from torch.nn import CosineSimilarity
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from pydub import AudioSegment


def convert_wav_to_mp3(wav_path, mp3_path):
    print(f"Converting {wav_path} to {mp3_path}")  # Debug info
    audio = AudioSegment.from_wav(wav_path)
    audio.export(mp3_path, format="mp3")
    print(f"Converted {wav_path} to {mp3_path}")  # Debug info


def calculate_model_similarity(path1, path2):
    state_dict1 = torch.load(path1)
    state_dict2 = torch.load(path2)

    if isinstance(state_dict1, torch.Tensor):
        params1 = state_dict1.flatten()
    elif isinstance(state_dict1, dict):
        params1 = flatten_parameters(state_dict1)
    else:
        raise ValueError("Unexpected format for state_dict1.")

    if isinstance(state_dict2, torch.Tensor):
        params2 = state_dict2.flatten()
    elif isinstance(state_dict2, dict):
        params2 = flatten_parameters(state_dict2)
    else:
        raise ValueError("Unexpected format for state_dict2.")

    if params1.shape != params2.shape:
        max_length = max(params1.shape[0], params2.shape[0])
        params1 = torch.nn.functional.pad(params1, (0, max_length - params1.shape[0]))
        params2 = torch.nn.functional.pad(params2, (0, max_length - params2.shape[0]))

    cos = CosineSimilarity(dim=0)
    similarity = cos(params1, params2)

    return similarity.item()


def flatten_parameters(state_dict):
    params = []
    for p in state_dict.values():
        if isinstance(p, torch.Tensor):
            if p.is_sparse:
                p = p.to_dense()
            params.append(p.flatten())
        elif hasattr(p, 'to_dense'):
            params.append(p.to_dense().flatten())
        else:
            print(f"Skipping non-tensor parameter: {p}")
    return torch.nn.utils.parameters_to_vector(params)


# Directory settings
embedding_dir = "E:\\UM Subject\\FinalProject\\dataset\\Embedding"
json_dir = "E:\\UM Subject\\FinalProject\\dataset\\MuVi-Sync\\dataset\\pretreat"
audio_dir = "E:\\UM Subject\\FinalProject\\dataset\\MuVi-Sync\\dataset\\basegen"

# Print all files in the audio directory with their extensions
print("Files in audio directory:")
for file in os.listdir(audio_dir):
    print(f"File: {file}, Extension: {os.path.splitext(file)[1]}")  # Debug info

# Store similarities
similarities = []

# Calculate similarities
for i in range(1, 295):
    embedding_filename = f"{i:03}_embeddings.pt"
    embedding_path = os.path.join(embedding_dir, embedding_filename)

    if os.path.isfile(embedding_path):
        similarity = calculate_model_similarity("E:\\UM Subject\\FinalProject\\dataset\\Embedding\\001_embeddings.pt",
                                                embedding_path)
        similarities.append((embedding_filename, similarity))
        print(f"Calculated similarity for {embedding_filename}: {similarity}")
    else:
        print(f"File {embedding_filename} does not exist. Skipping.")

# Get top 3 similarities
top_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:3]

# Prepare descriptions and melodies
descriptions = []
melodies = []

for filename, _ in top_similarities:
    # Adjusting to match the naming convention of JSON files
    json_index = filename[:3]  # Extracting the first three characters (e.g., '001', '070')
    json_path = os.path.join(json_dir, f"{json_index}.json")  # Constructing the JSON filename

    # Use the correct audio filename with three-digit format
    audio_filename = f"{json_index}_0.wav"
    audio_path = os.path.join(audio_dir, audio_filename)

    print(f"Trying to load audio file: {audio_path}")  # Debug info
    print(f"Absolute path: {os.path.abspath(audio_path)}")  # Debug info

    if os.path.isfile(json_path):
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)
            text_input = json.dumps(data)
            descriptions.append(text_input)
            print(f"Loaded description from {json_path}: {text_input}")  # Debug info
    else:
        print(f"[ERROR] JSON file not found: {json_path}")  # Debug info

    if os.path.isfile(audio_path):
        print(f"Found WAV file: {audio_path}")  # Debug info
        mp3_path = os.path.join(audio_dir, f"{audio_filename.replace('.wav', '.mp3')}")
        convert_wav_to_mp3(audio_path, mp3_path)  # Convert to MP3

        if os.path.isfile(mp3_path):  # Check if the conversion was successful
            print(f"Loading converted MP3 file: {mp3_path}")  # Debug info
            melody, sr = torchaudio.load(mp3_path)  # Load the MP3 file
            melodies.append(melody)
        else:
            print(f"Conversion to MP3 failed for {audio_path}")  # Debug info
    else:
        print(f"WAV file not found: {audio_path}")  # Debug info

# Ensure we have enough melodies and descriptions
if len(melodies) >= 3 and len(descriptions) >= 3:
    # 直接传递 3 个旋律和 3 个描述生成音乐
    model = MusicGen.get_pretrained('facebook/musicgen-melody')  # Use the full pre-trained id
    model.set_generation_params(duration=10)  # Generate 10 seconds of music

    # 使用描述和旋律生成音乐
    wav = model.generate_with_chroma(descriptions[:3], melodies[:3], sr)

    # Save generated music
    output_filename = 'generated_music.wav'
    audio_write(output_filename, wav[0].cpu(), model.sample_rate, strategy="loudness")
    print(f"Saved generated music: {output_filename}")
else:
    print("旋律或描述数量不足，无法生成音乐。")

print("音乐生成完成。")
import os
import json
import torch
import torchaudio
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from torch.nn import CosineSimilarity
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from pydub import AudioSegment

# Load processor and model
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

embedding_dir = "E:\\UM Subject\\FinalProject\\dataset\\Embedding"
json_dir = "E:\\UM Subject\\FinalProject\\dataset\\MuVi-Sync\\dataset\\pretreat"
audio_dir = "E:\\UM Subject\\FinalProject\\dataset\\MuVi-Sync\\dataset\\basegen"

def convert_wav_to_mp3(wav_path, mp3_path):
    audio = AudioSegment.from_wav(wav_path)
    audio.export(mp3_path, format="mp3")

def calculate_model_similarity(path1, path2):
    # Load the state dictionaries of the two models
    state_dict1 = torch.load(path1)
    state_dict2 = torch.load(path2)

    # Check if the loaded objects are tensors or dictionaries
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

    # Calculate cosine similarity
    cos = CosineSimilarity(dim=0)
    similarity = cos(params1, params2)

    return similarity.item()

# Helper function to flatten parameters in a dictionary
def flatten_parameters(state_dict):
    params = []
    for p in state_dict.values():
        if isinstance(p, torch.Tensor):
            params.append(p.flatten())
        elif hasattr(p, 'to_dense'):
            params.append(p.to_dense().flatten())
        else:
            print(f"Skipping non-tensor parameter: {p}")
    return torch.nn.utils.parameters_to_vector(params)
def generate_music_from_json(json_data):
    input_text = json.dumps(json_data)

    # Prepare input for the model
    inputs = processor(text=[input_text], padding=True, return_tensors="pt")
    input_ids = inputs["input_ids"].to('cuda')  # Move to GPU

    # Move the model to GPU
    model.to('cuda')  # Ensure the model is on the GPU

    # Get input embeddings
    with torch.no_grad():
        input_embeddings = model.get_input_embeddings()(input_ids)

    # Save the input embeddings temporarily
    input_embedding_path = "temp_input_embeddings.pt"
    torch.save(input_embeddings.cpu(), input_embedding_path)  # Save on CPU (optional)

    # Store similarities in a list
    similarities = []

    # Loop through the range of embeddings
    for i in range(1, 776):  # From 001 to 775
        embedding_filename = f"{i:03}_embeddings.pt"  # Format the number with leading zeros
        embedding_path = os.path.join(embedding_dir, embedding_filename)

        # Check if the embedding file exists
        if os.path.isfile(embedding_path):
            similarity = calculate_model_similarity(input_embedding_path, embedding_path)
            similarities.append((embedding_filename, similarity))
            print(f"Calculated similarity for {embedding_filename}: {similarity}")
        else:
            print(f"File {embedding_filename} does not exist. Skipping.")

    # Sort the similarities and get the top 3
    top_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:3]

    # Descriptions and melodies initialization
    descriptions = [input_text]  # Start with the input JSON as the first description
    melodies = [torch.zeros((1, 16000)).to('cuda')]  # Empty music tensor on GPU

    for filename, _ in top_similarities:
        json_index = filename[:3]
        audio_filename = f"{json_index}_0.wav"

        # Load corresponding JSON and audio file
        json_path = os.path.join(json_dir, f"{json_index}.json")
        audio_path = os.path.join(audio_dir, audio_filename)

        if os.path.isfile(json_path):
            with open(json_path, 'r') as json_file:
                data = json.load(json_file)
                descriptions.append(json.dumps(data))  # Add related descriptions

        if os.path.isfile(audio_path):
            mp3_path = os.path.join(audio_dir, f"{audio_filename.replace('.wav', '.mp3')}")
            convert_wav_to_mp3(audio_path, mp3_path)
            melody, sr = torchaudio.load(mp3_path)
            melodies.append(melody.to('cuda'))  # Ensure melody is on GPU

    # Ensure we have enough melodies and descriptions
    if len(melodies) == 4 and len(descriptions) == 4:  # Check for 4 each
        wav = model.generate_with_chroma(descriptions, melodies, sr)
        output_filename = 'g1.wav'
        audio_write(output_filename, wav[0].cpu(), model.sample_rate, strategy="loudness")  # Save on CPU
        print(f"Generated music saved as: {output_filename}")
    else:
        print("Insufficient melodies or descriptions to generate music.")

if __name__ == "__main__":
    # Example usage
    json_file_path = r"E:\UM Subject\FinalProject\dataset\MuVi-Sync\dataset\pretreat\001.json"  # Replace with your actual JSON file path
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)

    generate_music_from_json(json_data)
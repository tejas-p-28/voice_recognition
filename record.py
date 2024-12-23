import sounddevice as sd
import librosa
import numpy as np
import scipy.spatial.distance as distance
import ipywidgets as widgets
from IPython.display import display

# Function to record audio in real-time
def record_audio(duration, sr=16000):
    """Record real-time audio using the microphone."""
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait() 
    print(audio) # Wait for the recording to finish
    print("Recording finished.")
    return np.squeeze(audio), sr  # Return audio and the sample rate

# Function to extract MFCC features from audio
def extract_mfcc(audio, sr, n_mfcc=13):
    """Extract MFCC features from audio."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc

# Function to compare two MFCC vectors using Cosine Similarity
def compare_vectors(mfcc_real_time, mfcc_existing):
    """Compare two MFCC vectors using Cosine Similarity."""
    # Flatten the MFCC features to 1D vectors for comparison
    mfcc_real_time_flat = mfcc_real_time.flatten()
    mfcc_existing_flat = mfcc_existing.flatten()

    # Compute Cosine Similarity
    similarity = 1 - distance.cosine(mfcc_real_time_flat, mfcc_existing_flat)
    return similarity

# Load and process an existing audio sample
def process_existing_sample(file_path, sr=16000):
    """Load and process an existing audio sample."""
    # Load the existing audio file using librosa
    existing_audio, sr_existing = librosa.load(file_path, sr=sr)
    # Extract MFCC features from the existing sample
    mfcc_existing = extract_mfcc(existing_audio, sr_existing)
    return mfcc_existing, sr_existing

# Function to display audio and process it
def display_audio_and_compare(existing_audio_file):
    # Record real-time audio
    duration = int(input("Enter duration of recording in seconds: "))
    audio_real_time, sr_real_time = record_audio(duration)
    
    # Extract MFCC features from the real-time recording
    mfcc_real_time = extract_mfcc(audio_real_time, sr_real_time)

    # Display the recorded audio
    print(f"Audio shape: {audio_real_time.shape}, Sample rate: {sr_real_time}")
    
    # Load and process the existing audio sample
    mfcc_existing, sr_existing = process_existing_sample(existing_audio_file)
    
    # Compare the MFCC feature vectors (real-time vs existing)
    similarity = compare_vectors(mfcc_real_time, mfcc_existing)
    
    print(f"Cosine Similarity between the real-time audio and the existing sample: {similarity}")
    
    # Display the real-time audio in Colab
    display(widgets.Audio(value=audio_real_time, format='opus', rate=sr_real_time))

# Path to your existing audio file
existing_audio_file = "Train\1.m4a"  # Replace with the correct path to your sample

# Call the function to record and compare
display_audio_and_compare(existing_audio_file)

import whisper
import torch
import numpy as np
from datasets import load_dataset
from itertools import islice
import os
import json
from datetime import datetime
import pandas as pd
import soundfile as sf
from tqdm import tqdm

def calc_snr(original, delta):
    orginal_norm = torch.norm(original, p=2)
    delta_norm = torch.norm(delta, p=2)
    if delta_norm == 0:
        return float('inf')
    snr_db = 20 * (torch.log10(orginal_norm) - torch.log10(delta_norm))
    return snr_db.item()

def epsilon_from_snr(audio, target_snr_db):
    audio_norm = torch.norm(audio, p=2)
    epsilon = audio_norm / (10 ** (target_snr_db/20))
    return epsilon.item()

def pgd_attack_l2_original_whisper(whisper_model, raw_audio, target_snr_db, device, ground_truth_text=None, learning_rate_multiplier=0.1, n_steps=200):
    """
    Untargeted PGD attack using original OpenAI Whisper
    """
    # Prepare audio (pad/trim to 30 seconds and ensure float32)
    audio = whisper.pad_or_trim(raw_audio.astype(np.float32))
    torch_audio = torch.from_numpy(audio).to(device).float()
    
    epsilon = epsilon_from_snr(torch_audio, target_snr_db)
    learning_rate = learning_rate_multiplier * epsilon
    
    delta = torch.zeros_like(torch_audio, requires_grad=True, device=device)
    
    with torch.no_grad():
        def get_model_transcription():
            clean_mel = whisper.log_mel_spectrogram(torch_audio, n_mels=whisper_model.dims.n_mels).to(device)
            options = whisper.DecodingOptions(language='en', without_timestamps=True)
            result = whisper.decode(whisper_model, clean_mel, options)
            return result.text
        
        if ground_truth_text is None or len(ground_truth_text.strip()) == 0:
            print("No valid ground truth provided. Generating model transcription...")
            ground_truth_text = get_model_transcription()
            print(f"Generated transcription: {ground_truth_text}")
        
        tokenizer = whisper.tokenizer.get_tokenizer(whisper_model.is_multilingual, language='en')
        try:
            ground_truth_tokens = tokenizer.encode(ground_truth_text)
            print("Using provided ground truth tokens")
            if len(ground_truth_tokens) == 0:
                raise ValueError("Tokenization resulted in empty token sequence")
        except Exception as e:
            print(f"Tokenization failed for: '{ground_truth_text}'")
            print(f"Error: {e}")
            print("Falling back to model transcription...")
            ground_truth_text = get_model_transcription()
            ground_truth_tokens = tokenizer.encode(ground_truth_text)
        
        ground_truth_tokens = torch.tensor(ground_truth_tokens, device=device).unsqueeze(0)
    
    print(f"Ground truth: {ground_truth_text}")
    print(f"Ground truth tokens shape: {ground_truth_tokens.shape}")
    
    for step in range(n_steps):
        if delta.grad is not None:
            delta.grad.zero_()
            
        perturbed_audio = torch_audio + delta
        mel_spec = whisper.log_mel_spectrogram(
            perturbed_audio, 
            n_mels=whisper_model.dims.n_mels
        ).to(device)
        
        encoder_output = whisper_model.encoder(mel_spec.unsqueeze(0))
        
        if ground_truth_tokens.size(1) > 1:
            decoder_input = ground_truth_tokens[:, :-1]
            decoder_target = ground_truth_tokens[:, 1:]
            
            decoder_output = whisper_model.decoder(decoder_input, encoder_output)
            logits = decoder_output
            
            # Compute cross entropy loss
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), 
                decoder_target.reshape(-1),
                ignore_index=-100
            )
            
        else:
            loss = torch.mean(encoder_output**2)

        loss.backward()
        
        with torch.no_grad():
            if delta.grad is not None:
                grad = delta.grad
                grad_norm = torch.norm(grad, p=2)
                if grad_norm > 0:
                    delta.data = delta.data + (grad / grad_norm) * learning_rate
                    
                    # Project to epsilon ball
                    delta_norm = torch.norm(delta.data, p=2)
                    if delta_norm > epsilon:
                        delta.data = (delta.data / delta_norm) * epsilon
        
        if ((step+1) % 20) == 0:
            print(f"Step {step+1}/{n_steps} || Loss: {loss.item():.4f}")
    
    with torch.no_grad():
        adversarial_audio_full = (torch_audio + delta).cpu().numpy()
        adversarial_audio = adversarial_audio_full[:len(raw_audio)]
        final_snr = calc_snr(torch.from_numpy(raw_audio.astype(np.float32)), delta[:len(raw_audio)].cpu())
    
    return adversarial_audio, final_snr

# Complete setup and test
def run_original_whisper_attack():
    """
    Complete example using original Whisper
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load original Whisper model
    print("Loading original Whisper model...")
    whisper_model = whisper.load_model("base.en").to(device)
    print("Model loaded successfully!")

    '''
    # Load sample audio (same as your setup)
    print("Loading sample audio...")
    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    sample_number = 1
    sample = dataset[sample_number]
    audio_array = sample["audio"]["array"]
    ground_truth_transcription = sample["text"]
    print(f"Original transcription: {ground_truth_transcription}")
    print(f"Raw audio shape: {audio_array.shape}")
    '''

    # Load sample data - using LibriSpeech as in the paper
    print("Loading sample audio...")
    dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
    sample_number=12    
    sample = next(islice(dataset, sample_number, sample_number+1))
    
    audio_array = sample["audio"]["array"]
    sampling_rate = sample["audio"]["sampling_rate"] 
    ground_truth_transcription = sample["text"]
    
    print(f"Ground truth: {ground_truth_transcription}")
    
    # Run attack
    print("\n### Running L2 PGD Attack with Original Whisper ###")
    adversarial_waveform, final_snr = pgd_attack_l2_original_whisper(
        whisper_model=whisper_model,
        raw_audio=audio_array,
        target_snr_db=35,
        device=device,
        ground_truth_text=ground_truth_transcription,
        n_steps=300
    )
    
    print(f"Final SNR: {final_snr}dB")
    
    # Test the adversarial audio
    print("\n### Testing Adversarial Audio ###")
    adversarial_audio_padded = whisper.pad_or_trim(adversarial_waveform.astype(np.float32))
    adversarial_mel = whisper.log_mel_spectrogram(
        torch.from_numpy(adversarial_audio_padded).to(device), 
        n_mels=whisper_model.dims.n_mels
    ).to(device)
    
    options = whisper.DecodingOptions(language='en', without_timestamps=True)
    adversarial_result = whisper.decode(whisper_model, adversarial_mel, options)
    
    print(f"Original transcription: {ground_truth_transcription}")
    print(f"Adversarial transcription: {adversarial_result.text}")
    
    # Save audio file
    import soundfile as sf
    
    
    sf.write(f"normal_whisper_{sample_number}.wav", audio_array, 16000)
    print(f"Normal audio saved as 'normal_whisper_{sample_number}.wav'")
    Audio(f"normal_whisper_{sample_number}.wav")

    
    sf.write(f"adversarial_original_whisper_{sample_number}.wav", adversarial_waveform, 16000)
    print(f"Adversarial audio saved as 'adversarial_original_whisper_{sample_number}.wav'")
    Audio(f"adversarial_original_whisper_{sample_number}.wav")
    
    return adversarial_waveform, final_snr

def run_batch_whisper_attack(
    model_name="tiny.en",
    dataset_name="librispeech_asr",
    dataset_config="clean", 
    dataset_split="test",
    num_samples=10,
    start_idx=0,
    target_snr_db=35,
    n_steps=300,
    output_dir="whisper_attacks"
):
    """
    Run adversarial attacks on multiple samples and save results organized by folders
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(output_dir, f"attack_{timestamp}")
    
    # Create subdirectories
    audio_dir = os.path.join(experiment_dir, "audio")
    results_dir = os.path.join(experiment_dir, "results")
    
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Output directory: {experiment_dir}")
    
    # Load model once
    print("Loading original Whisper model...")
    whisper_model = whisper.load_model(model_name).to(device)
    print("Model loaded successfully!")
    
    # Load dataset
    print(f"Loading {dataset_name} dataset...")
    if dataset_name == "librispeech_asr":
        dataset = load_dataset(dataset_name, dataset_config, split=dataset_split, streaming=True)
    else:
        dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
    
    # Storage for results
    results = []
    failed_samples = []
    
    # Process samples in batch
    print(f"\nProcessing {num_samples} samples starting from index {start_idx}...")
    
    if hasattr(dataset, '__iter__'):  # Streaming dataset
        samples = list(islice(dataset, start_idx, start_idx + num_samples))
    else:  # Regular dataset
        samples = dataset.select(range(start_idx, min(start_idx + num_samples, len(dataset))))
    
    for i, sample in enumerate(tqdm(samples, desc="Running attacks")):
        sample_idx = start_idx + i
        
        try:
            # Extract sample data
            audio_array = sample["audio"]["array"]
            ground_truth_text = sample["text"]
            
            print(f"\nSample {sample_idx}: {ground_truth_text[:50]}...")
            
            # Run attack
            adversarial_waveform, final_snr = pgd_attack_l2_original_whisper(
                whisper_model=whisper_model,
                raw_audio=audio_array,
                target_snr_db=target_snr_db,
                device=device,
                #ground_truth_text=ground_truth_text,
                n_steps=n_steps
            )
            
            # Test adversarial result
            adversarial_audio_padded = whisper.pad_or_trim(adversarial_waveform.astype(np.float32))
            adversarial_mel = whisper.log_mel_spectrogram(
                torch.from_numpy(adversarial_audio_padded).to(device), 
                n_mels=whisper_model.dims.n_mels
            ).to(device)
            
            options = whisper.DecodingOptions(language='en', without_timestamps=True)
            adversarial_result = whisper.decode(whisper_model, adversarial_mel, options)
            adversarial_text = adversarial_result.text
            
            # Calculate attack success (simple metric: transcription changed)
            attack_success = ground_truth_text.lower().strip() != adversarial_text.lower().strip()
            
            # Save audio files
            original_filename = os.path.join(audio_dir, f"original_{sample_idx:04d}.wav")
            adversarial_filename = os.path.join(audio_dir, f"adversarial_{sample_idx:04d}.wav")
            
            sf.write(original_filename, audio_array, 16000)
            sf.write(adversarial_filename, adversarial_waveform, 16000)
            
            # Store results
            result = {
                "sample_idx": sample_idx,
                "original_transcription": ground_truth_text,
                "adversarial_transcription": adversarial_text,
                "final_snr_db": final_snr,
                "target_snr_db": target_snr_db,
                "attack_success": attack_success,
                "original_audio_file": original_filename,
                "adversarial_audio_file": adversarial_filename,
                "audio_length_seconds": len(audio_array) / 16000
            }
            
            results.append(result)
            
            print(f"SNR: {final_snr:.2f}dB, Success: {attack_success}")
            print(f"Original: {ground_truth_text}")
            print(f"Adversarial: {adversarial_text}")
            
        except Exception as e:
            print(f"Failed on sample {sample_idx}: {str(e)}")
            failed_samples.append({"sample_idx": sample_idx, "error": str(e)})
            continue
    
    # Save results
    print(f"\nSaving results to {results_dir}...")
    
    # Save detailed results as JSON
    results_file = os.path.join(results_dir, "attack_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            "experiment_config": {
                "model": model_name,
                "dataset": dataset_name,
                "dataset_config": dataset_config,
                "dataset_split": dataset_split,
                "num_samples": num_samples,
                "start_idx": start_idx,
                "target_snr_db": target_snr_db,
                "n_steps": n_steps,
                "timestamp": timestamp
            },
            "results": results,
            "failed_samples": failed_samples
        }, f, indent=2)
    
    # Save summary as CSV
    if results:
        df = pd.DataFrame(results)
        summary_file = os.path.join(results_dir, "attack_summary.csv")
        df.to_csv(summary_file, index=False)
        
        # Print summary statistics
        print("\n" + "="*50)
        print("ATTACK SUMMARY")
        print("="*50)
        print(f"Total samples processed: {len(results)}")
        print(f"Attack success rate: {df['attack_success'].mean()*100:.1f}%")
        print(f"Average SNR: {df['final_snr_db'].mean():.2f} ± {df['final_snr_db'].std():.2f} dB")
        print(f"Failed samples: {len(failed_samples)}")
        print(f"Results saved to: {experiment_dir}")
        
        return results, experiment_dir
    else:
        print("No successful attacks to summarize.")
        return [], experiment_dir


def analyze_attack_results(experiment_dir):
    """
    Load and analyze results from a previous attack experiment
    """
    results_file = os.path.join(experiment_dir, "results", "attack_results.json")
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data["results"]
    config = data["experiment_config"]
    
    print("EXPERIMENT CONFIGURATION:")
    print(f"Dataset: {config['dataset']}")
    print(f"Samples: {config['num_samples']}")
    print(f"Target SNR: {config['target_snr_db']}dB")
    print(f"Steps: {config['n_steps']}")
    
    if results:
        df = pd.DataFrame(results)
        
        print("\nRESULTS ANALYSIS:")
        print(f"Success rate: {df['attack_success'].mean()*100:.1f}%")
        print(f"Average achieved SNR: {df['final_snr_db'].mean():.2f}dB")
        print(f"SNR std: {df['final_snr_db'].std():.2f}dB")
        print(f"Audio length range: {df['audio_length_seconds'].min():.1f}s - {df['audio_length_seconds'].max():.1f}s")
        
        # Show some examples
        print(f"\nEXAMPLE SUCCESSFUL ATTACKS:")
        successful = df[df['attack_success']]
        for _, row in successful.head(3).iterrows():
            print(f"Sample {row['sample_idx']}:")
            print(f"  Original: {row['original_transcription'][:80]}...")
            print(f"  Adversarial: {row['adversarial_transcription'][:80]}...")
            print(f"  SNR: {row['final_snr_db']:.2f}dB")
            print()

if __name__ == "__main__":
    run_batch_whisper_attack(
    model_name="tiny",
    dataset_name="librispeech_asr",
    dataset_config="clean", 
    dataset_split="test",
    num_samples=20,
    start_idx=10,
    target_snr_db=35,
    n_steps=200,
    output_dir="whisper_attacks"
)
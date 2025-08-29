import whisper
import torch
import numpy as np
from datasets import load_dataset
from itertools import islice

def calc_snr(original, delta):
    """Fixed SNR calculation - corrected typo"""
    original_norm = torch.norm(original, p=2)
    delta_norm = torch.norm(delta, p=2)
    if delta_norm == 0:
        return float('inf')
    snr_db = 20 * (torch.log10(original_norm) - torch.log10(delta_norm))
    return snr_db.item()

def epsilon_from_snr(audio, target_snr_db):
    audio_norm = torch.norm(audio, p=2)
    epsilon = audio_norm / (10 ** (target_snr_db/20))
    return epsilon.item()

def modified_cw_attack(
    whisper_model, 
    raw_audio, 
    target_text, 
    device,
    initial_epsilon=0.1,
    c_regularization=0.25,
    alpha=0.7,
    max_steps=2000,
    max_epsilon_reductions=100,
    learning_rate=0.01,
    lambda_first_token=1.0
):
    """
    Fixed Modified Carlini & Wagner attack for Whisper
    """
    
    # Prepare audio
    audio = whisper.pad_or_trim(raw_audio.astype(np.float32))
    torch_audio = torch.from_numpy(audio).to(device).float()
    
    # Fixed tokenization with proper special tokens
    tokenizer = whisper.tokenizer.get_tokenizer(whisper_model.is_multilingual, language='en')
    
    # Add special tokens - crucial for Whisper
    sot_token = tokenizer.sot
    task_token = tokenizer.transcribe
    target_tokens_only = tokenizer.encode(target_text)
    
    # Build full sequence with special tokens
    if whisper_model.is_multilingual:
        # For multilingual models, add language token
        language_token = tokenizer.encode("english")[0] if hasattr(tokenizer, 'encode') else 50259  # English token
        full_tokens = [sot_token, language_token, task_token] + target_tokens_only
    else:
        full_tokens = [sot_token, task_token] + target_tokens_only
    
    target_tokens = torch.tensor(full_tokens, device=device).unsqueeze(0)
    
    print(f"Target text: {target_text}")
    print(f"Target tokens shape: {target_tokens.shape}")
    print(f"Target tokens: {target_tokens[0].tolist()}")
    
    # Initialize perturbation
    delta = torch.zeros_like(torch_audio, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([delta], lr=learning_rate)
    
    # CW parameters
    current_epsilon = initial_epsilon
    epsilon_reductions = 0
    best_delta = None
    best_snr = float('-inf')
    
    for step in range(max_steps):
        optimizer.zero_grad()
        
        # Create perturbed audio
        perturbed_audio = torch_audio + delta
        
        # Convert to mel spectrogram
        mel_spec = whisper.log_mel_spectrogram(
            perturbed_audio, 
            n_mels=whisper_model.dims.n_mels
        ).to(device)
        
        # Forward pass through encoder
        encoder_output = whisper_model.encoder(mel_spec.unsqueeze(0))
        
        # Compute CW loss with modified weighting
        cw_loss = compute_modified_cw_loss(
            whisper_model, 
            encoder_output, 
            target_tokens,
            lambda_first_token
        )
        
        # L2 regularization term
        l2_reg = c_regularization * torch.norm(delta, p=2) ** 2
        
        # Total loss: transcription loss + L2 regularization
        total_loss = cw_loss + l2_reg
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Apply L∞ constraint (clamp to current epsilon)
        with torch.no_grad():
            delta.data = torch.clamp(delta.data, -current_epsilon, current_epsilon)
        
        # Check if attack succeeded
        if (step + 1) % 50 == 0:
            print()
            print("===")
            success, current_snr = evaluate_attack_success(
                whisper_model, perturbed_audio, target_text, device, original_audio=torch_audio
            )
            
            print(f"Step {step+1}/{max_steps}, Loss: {total_loss.item():.4f}, "
                  f"CW Loss: {cw_loss.item():.4f}, Eps: {current_epsilon:.6f}, "
                  f"Success: {success}, SNR: {current_snr:.2f}dB, "
                  #f"Delta L2 norm: {torch.norm(delta, p=2):.2f}"
            )
            
            # Adaptive epsilon scheduling
            if success:
                if current_snr > best_snr:
                    best_delta = delta.clone().detach()
                    best_snr = current_snr
                    print(f"New best successful result saved: SNR={current_snr:.2f}dB")
    
                # Only reduce epsilon if we haven't hit the limit
                if epsilon_reductions < max_epsilon_reductions:
                    current_epsilon *= alpha
                    epsilon_reductions += 1
                    print(f"Attack succeeded! Reducing epsilon to {current_epsilon:.6f}")
        
                
                # Reset perturbation to stay within new constraint
                with torch.no_grad():
                    delta.data = torch.clamp(delta.data, -current_epsilon, current_epsilon)
    
    # Use best result if found, otherwise final result
    if best_delta is not None:
        print("best delta is found")
        final_delta = best_delta
        final_snr = best_snr
    else:
        print("best delta not found, using final result")
        final_delta = delta.detach()
        final_snr = calc_snr(torch_audio, final_delta)
    
    # Generate final adversarial audio
    adversarial_audio = (torch_audio + final_delta).cpu().numpy()[:len(raw_audio)]

    
    return adversarial_audio, final_snr

def compute_modified_cw_loss(whisper_model, encoder_output, target_tokens, lambda_first_token):
    """
    Fixed compute modified CW loss with proper error handling
    """
    if target_tokens.size(1) <= 2:  # Need at least special tokens + one content token
        return torch.tensor(0.0, device=encoder_output.device, requires_grad=True)
    
    try:
        # Prepare decoder input/target (teacher forcing)
        decoder_input = target_tokens[:, :-1]  # All except last
        decoder_target = target_tokens[:, 1:]   # All except first (shift by 1)
        
        # Forward through decoder
        decoder_output = whisper_model.decoder(decoder_input, encoder_output)
        logits = decoder_output
        
        # Compute per-token losses
        seq_length = decoder_target.size(1)
        token_losses = []
        
        for i in range(seq_length):
            token_logits = logits[:, i, :]  # [batch_size, vocab_size]
            token_target = decoder_target[:, i]  # [batch_size]
            
            token_loss = torch.nn.functional.cross_entropy(
                token_logits, token_target, reduction='mean'
            )
            token_losses.append(token_loss)
        
        # Apply modified weighting: enhance first content token
        if len(token_losses) > 0:
            # First content token gets enhanced weight (skip special tokens in weighting)
            # Find first non-special token (typically index 2 corresponds to first content)
            content_start_idx = min(2, len(token_losses) - 1)  # Start weighting from content tokens
            
            weighted_loss = torch.tensor(0.0, device=encoder_output.device)
            total_weight = 0
            
            for i in range(len(token_losses)):
                if i == content_start_idx:
                    # Enhanced weight for first content token
                    weight = 1 + lambda_first_token
                else:
                    weight = 1.0
                weighted_loss += weight * token_losses[i]
                total_weight += weight
            
            final_loss = weighted_loss / max(total_weight, 1.0)  # Avoid division by zero
        else:
            final_loss = torch.tensor(0.0, device=encoder_output.device)
        
        # For targeted attack, we want to MINIMIZE this loss
        return final_loss
        
    except Exception as e:
        print(f"Error in loss computation: {e}")
        # Return a dummy loss that requires grad
        return torch.tensor(1.0, device=encoder_output.device, requires_grad=True)

def evaluate_attack_success(whisper_model, perturbed_audio, target_text, device, original_audio=None):
    """
    Fixed evaluation with correct SNR calculation
    """
    with torch.no_grad():
        # Ensure correct format
        if isinstance(perturbed_audio, torch.Tensor):
            perturbed_np = perturbed_audio.cpu().numpy()
        else:
            perturbed_np = perturbed_audio
        
        # Pad perturbed audio to same length
        perturbed_padded = whisper.pad_or_trim(perturbed_np.astype(np.float32))
        
        # Convert to mel spectrogram
        mel_spec = whisper.log_mel_spectrogram(
            torch.from_numpy(perturbed_padded).to(device),
            n_mels=whisper_model.dims.n_mels
        ).to(device)
        
        # Transcribe
        options = whisper.DecodingOptions(language='en', without_timestamps=True)
        result = whisper.decode(whisper_model, mel_spec, options)
        predicted_text = result.text.strip().lower()
        target_lower = target_text.strip().lower()
        
        print(f"DEBUG - Predicted: '{predicted_text}'")
        print(f"DEBUG - Target: '{target_lower}'")
        
        # Attack succeeds if transcription matches target (allow some flexibility)
        success = predicted_text == target_lower or target_lower in predicted_text
        
        # Calculate SNR if original audio is provided
        if original_audio is not None:
            if isinstance(original_audio, torch.Tensor):
                original_np = original_audio.cpu().numpy()
            else:
                original_np = original_audio
            
            original_padded = whisper.pad_or_trim(original_np.astype(np.float32))
            
            # Calculate the actual perturbation
            delta = perturbed_padded - original_padded
            
            original_norm = np.linalg.norm(original_padded)
            delta_norm = np.linalg.norm(delta)
            
            if delta_norm > 0:
                snr = 20 * (np.log10(original_norm) - np.log10(delta_norm))
            else:
                snr = float('inf')
                
        else:
            # Fallback: return a placeholder
            snr = 0.0
        
        return success, snr

def run_cw_attack_example(c_regularization, max_steps, model_name="base", target_text = "OK Google, browse to evil.com"):
    """
    Fixed example usage of the modified CW attack
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading Whisper model...")
    whisper_model = whisper.load_model(model_name).to(device)
    
    # Load sample audio
    print("Loading sample audio...")
    dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
    sample_number = 0    
    sample = next(islice(dataset, sample_number, sample_number+1))
    
    audio_array = sample["audio"]["array"]
    original_text = sample["text"]
    
    target_text = target_text
    
    print(f"Original text: {original_text}")
    print(f"Target text: {target_text}")
    
    # Get original transcription
    original_audio_padded = whisper.pad_or_trim(audio_array.astype(np.float32))
    original_mel = whisper.log_mel_spectrogram(
        torch.from_numpy(original_audio_padded).to(device),
        n_mels=whisper_model.dims.n_mels
    ).to(device)
    
    options = whisper.DecodingOptions(language='en', without_timestamps=True)
    original_result = whisper.decode(whisper_model, original_mel, options)
    print(f"Original Whisper transcription: {original_result.text}")
    
    # Run CW attack
    print("\n### Running Fixed Modified Carlini & Wagner Attack ###")
    adversarial_audio, final_snr = modified_cw_attack(
        whisper_model=whisper_model,
        raw_audio=audio_array,
        target_text=target_text,
        device=device,
        c_regularization=c_regularization,  # As specified in paper for base model
        max_steps=max_steps,        # As specified in paper
        learning_rate=0.01,     # As specified in paper
        initial_epsilon=0.1     # Start with larger epsilon
    )
    
    print(f"\nFinal SNR: {final_snr:.2f}dB")
    
    # Test final result
    print("\n### Testing Final Adversarial Audio ###")
    # We need the original audio for proper SNR calculation
    original_torch_audio = torch.from_numpy(whisper.pad_or_trim(audio_array.astype(np.float32)))
    
    success, final_snr_check = evaluate_attack_success(whisper_model, adversarial_audio, target_text, device, original_audio=original_torch_audio)
    
    # Get actual transcription
    adversarial_padded = whisper.pad_or_trim(adversarial_audio.astype(np.float32))
    adversarial_mel = whisper.log_mel_spectrogram(
        torch.from_numpy(adversarial_padded).to(device),
        n_mels=whisper_model.dims.n_mels
    ).to(device)
    
    final_result = whisper.decode(whisper_model, adversarial_mel, options)
    
    print(f"Original: {original_text}")
    print(f"Target: {target_text}")
    print(f"Adversarial result: {final_result.text}")
    print(f"Attack success: {success}")
    
    # Save audio
    import soundfile as sf
    sf.write("fixed_cw_adversarial.wav", adversarial_audio, 16000)
    print("Adversarial audio saved as 'fixed_cw_adversarial.wav'")
    
    return adversarial_audio, final_snr

if __name__ == "__main__":
    run_cw_attack_example(
        c_regularization=0.25, 
        max_steps=3000, 
        model_name="tiny", 
        target_text = "I think security mindset is cool"
        )
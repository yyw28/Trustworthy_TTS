#!/usr/bin/env python3
"""
Train TTS model with RL using TIS dataset.

This script performs two main steps:

STEP 1: Prepare TTS training data
    - Extracts text transcripts from TIS audio files using Whisper ASR (--use_asr)
    - Creates CSV files: train.csv, val.csv, test.csv
    - Format: wav_path|transcript_text|speaker_id
    - Skips this step if CSV files already exist

STEP 2: Train TTS model with Reinforcement Learning
    - Trains Tacotron2 TTS model
    - Uses RL policy to optimize GST weights for trustworthiness
    - Uses HuBERT classifier to provide trustworthiness rewards

Usage:
    python run_tts_rl_with_tis.py \
        --tis_dir /path/to/tis \
        --hubert_checkpoint /path/to/hubert/checkpoint.ckpt \
        --use_asr
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train TTS with RL using TIS data")
    
    # TIS data arguments
    parser.add_argument("--tis_dir", type=str, required=True, help="Path to TIS dataset directory")
    parser.add_argument("--use_asr", action="store_true", help="Use Whisper ASR to extract transcripts")
    
    # HuBERT checkpoint
    parser.add_argument("--hubert_checkpoint", type=str, required=True, help="Path to trained HuBERT checkpoint")
    
    # Tacotron2 checkpoint (pre-trained Tacotron2 + GST)
    parser.add_argument("--tacotron2_checkpoint", type=str, default=None, help="Path to pre-trained Tacotron2 + GST checkpoint (optional)")
    
    # TTS training arguments (passed to run_tts_with_rl.py)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loader workers")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum epochs")
    parser.add_argument("--rl_temperature", type=float, default=1.0, help="RL sampling temperature")
    parser.add_argument("--rl_entropy_coef", type=float, default=0.01, help="RL entropy coefficient")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--accelerator", type=str, default="gpu", choices=["gpu", "cpu", "mps"])
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    
    # Output directory for TTS CSV files (contains train.csv, val.csv, test.csv)
    parser.add_argument("--tts_csv_dir", type=str, default="./tis_tts_data", help="Directory for TTS CSV files")
    
    # Audio saving arguments
    parser.add_argument("--save_audio_dir", type=str, default=None, help="Directory to save generated audio files during training (optional)")
    parser.add_argument("--save_audio_every_n_steps", type=int, default=100, help="Save audio every N training steps (default: 100)")
    
    args = parser.parse_args()
    
    # ============================================================================
    # STEP 1: Prepare TTS training data (extract transcripts and create CSV files)
    # ============================================================================
    # This step converts TIS audio files into TTS training format:
    # - Extracts text transcripts using ASR
    # - Creates CSV files: train.csv, val.csv, test.csv
    # - Format: wav_path|transcript_text|speaker_id
    # ============================================================================
    
    tis_dir = Path(args.tis_dir)
    tts_csv_dir = Path(args.tts_csv_dir)
    train_csv = tts_csv_dir / "train.csv"
    val_csv = tts_csv_dir / "val.csv"
    
    # Check if CSV files already exist (skip if already prepared)
    if not train_csv.exists() or not val_csv.exists():
        print("=" * 80)
        print("STEP 1: Preparing TTS training data")
        print("=" * 80)
        print("Extracting transcripts and creating CSV files...")
        print()
        
        # Find the data preparation script
        prepare_script = Path(__file__).parent / "prepare_tis_for_tts.py"
        if not prepare_script.exists():
            raise FileNotFoundError(f"prepare_tis_for_tts.py not found: {prepare_script}")
        
        # Build command to run the preparation script
        cmd = [
            sys.executable,
            str(prepare_script),
            "--tis_dir", str(tis_dir),
            "--output_dir", str(tts_csv_dir),
        ]
        
        # Use Whisper ASR to extract real transcripts from audio
        if args.use_asr:
            cmd.append("--use_asr")
            print("  Using Whisper ASR to extract transcripts from audio files")
        
        print(f"\nRunning: {' '.join(cmd)}")
        print()
        
        # Run the preparation script
        result = subprocess.run(cmd, check=True)
        
        # Verify CSV files were created successfully
        if not train_csv.exists() or not val_csv.exists():
            raise RuntimeError(
                f"Failed to create TTS CSV files in {tts_csv_dir}\n"
                f"Expected files: {train_csv}, {val_csv}"
            )
        
        print("\n✓ TTS CSV files created successfully")
        print(f"  - Training data: {train_csv}")
        print(f"  - Validation data: {val_csv}")
    else:
        print("=" * 80)
        print("STEP 1: Using existing TTS training data")
        print("=" * 80)
        print(f"✓ Found existing CSV files in {tts_csv_dir}")
        print(f"  - Training data: {train_csv}")
        print(f"  - Validation data: {val_csv}")
        print("  (Skipping data preparation step)")
    
    # ============================================================================
    # STEP 2: Train TTS model with Reinforcement Learning
    # ============================================================================
    # This step trains the TTS model using:
    # - Tacotron2 for mel spectrogram generation
    # - RL policy to optimize GST weights for trustworthiness
    # - HuBERT classifier to provide trustworthiness rewards
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("STEP 2: Training TTS model with Reinforcement Learning")
    print("=" * 80)
    
    run_tts_script = Path(__file__).parent / "run_tts_with_rl.py"
    if not run_tts_script.exists():
        raise FileNotFoundError(f"run_tts_with_rl.py not found: {run_tts_script}")
    
    cmd = [
        sys.executable,
        str(run_tts_script),
        "--tts_data_dir", str(tis_dir.absolute()),
        "--train_csv", str(train_csv.absolute()),
        "--val_csv", str(val_csv.absolute()),
        "--hubert_checkpoint", args.hubert_checkpoint if args.hubert_checkpoint else "",
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--max_epochs", str(args.max_epochs),
        "--rl_temperature", str(args.rl_temperature),
        "--rl_entropy_coef", str(args.rl_entropy_coef),
        "--learning_rate", str(args.learning_rate),
        "--accelerator", args.accelerator,
        "--devices", str(args.devices),
    ]
    
    # Add Tacotron2 checkpoint if provided
    if args.tacotron2_checkpoint:
        cmd.extend(["--tacotron2_checkpoint", args.tacotron2_checkpoint])
        print(f"  Tacotron2 checkpoint: {args.tacotron2_checkpoint}")
    
    # Add audio saving arguments if specified
    if args.save_audio_dir:
        cmd.extend(["--save_audio_dir", args.save_audio_dir])
        cmd.extend(["--save_audio_every_n_steps", str(args.save_audio_every_n_steps)])
        print(f"  Audio saving enabled: {args.save_audio_dir} (every {args.save_audio_every_n_steps} steps)")
    
    print(f"Running: {' '.join(cmd)}")
    print("\n" + "=" * 80)
    
    result = subprocess.run(cmd, check=True)
    
    print("\n" + "=" * 80)
    print("TTS training with RL complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

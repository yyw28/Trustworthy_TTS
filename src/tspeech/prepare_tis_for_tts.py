#!/usr/bin/env python3
"""
Convert TIS dataset to TTS training format.

This script:
1. Reads TIS Speech_dataset_characteristics.csv
2. Extracts text transcripts (using ASR or placeholder)
3. Creates TTS CSV files (train.csv, val.csv, test.csv) with format: wav|text|speaker_idx
4. Optionally uses Whisper ASR to extract real transcripts

Usage:
    # Option 1: Use ASR to extract transcripts (recommended)
    python prepare_tis_for_tts.py \
        --tis_dir /path/to/tis \
        --output_dir ./tis_tts_data \
        --use_asr
    
    # Option 2: Use placeholder text (for testing)
    python prepare_tis_for_tts.py \
        --tis_dir /path/to/tis \
        --output_dir ./tis_tts_data \
        --placeholder_text "This is a trustworthy statement."
"""

import argparse
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def extract_transcript_with_asr(audio_path: str, asr_model=None, use_whisper: bool = True) -> str:
    """Extract transcript from audio using ASR."""
    if use_whisper:
        try:
            if asr_model is None:
                import whisper
                asr_model = whisper.load_model("base")
            result = asr_model.transcribe(audio_path)
            return result["text"].strip()
        except ImportError:
            print("Warning: whisper not installed. Install with: pip install openai-whisper")
            return None
        except Exception as e:
            print(f"Warning: ASR failed for {audio_path}: {e}")
            return None
    else:
        # Could use other ASR libraries here
        return None


def create_tts_csv_from_tis(
    tis_dir: str,
    output_dir: str,
    use_asr: bool = False,
    placeholder_text: str = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
):
    """
    Convert TIS dataset to TTS format.
    
    Parameters
    ----------
    tis_dir : str
        Path to TIS dataset directory
    output_dir : str
        Output directory for TTS CSV files
    use_asr : bool
        Whether to use ASR to extract transcripts (requires whisper)
    placeholder_text : str
        Placeholder text to use if ASR is not available
    train_ratio : float
        Ratio for training set
    val_ratio : float
        Ratio for validation set
    test_ratio : float
        Ratio for test set
    """
    tis_dir = Path(tis_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read TIS CSV
    csv_path = tis_dir / "Speech_dataset_characteristics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"TIS CSV not found: {csv_path}")
    
    print(f"Reading TIS CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Find available audio files
    wav_files = set()
    speech_wav_dir = tis_dir / "Speech WAV Files"
    if not speech_wav_dir.exists():
        raise FileNotFoundError(f"Speech WAV Files directory not found: {speech_wav_dir}")
    
    for root, dirs, files in os.walk(speech_wav_dir):
        for file in files:
            if file.endswith(".wav"):
                wav_files.add(file.split(".")[0])
    
    print(f"Found {len(wav_files)} audio files")
    
    # Filter dataframe to only include available files
    df["Audio_Filename"] = df["Audio_Filename"].str.strip()
    df = df[df["Audio_Filename"].isin(wav_files)]
    print(f"Found {len(df)} matching entries in CSV")
    
    # Prepare TTS data
    tts_data = []
    
    print("\nProcessing audio files...")
    if use_asr:
        print("Using ASR to extract transcripts (this may take a while)...")
        try:
            import whisper
            print("Loading Whisper model...")
            asr_model = whisper.load_model("base")
        except ImportError:
            print("Warning: whisper not installed. Falling back to placeholder text.")
            use_asr = False
            asr_model = None
    else:
        asr_model = None
    
    for idx, row in df.iterrows():
        # Construct audio file path
        ethnicity = row['Speaker_Ethnicity'].replace('_', ' ')
        age_group = row['Speaker_AgeGroup']
        audio_filename = row['Audio_Filename'].strip()
        
        audio_path = speech_wav_dir / f"{ethnicity} {age_group}" / f"{audio_filename}.wav"
        
        if not audio_path.exists():
            continue
        
        # Get transcript
        if use_asr and asr_model:
            transcript = extract_transcript_with_asr(str(audio_path), asr_model=asr_model, use_whisper=True)
            if transcript is None or len(transcript.strip()) == 0:
                transcript = placeholder_text or "This is a trustworthy statement."
        elif placeholder_text:
            transcript = placeholder_text
        else:
            # Use a generic transcript based on trustworthiness
            if row.get("Speaker_Intent") == "Trustworthy":
                transcript = "This is a trustworthy and reliable statement."
            else:
                transcript = "This is a statement."
        
        # Create relative path for CSV (relative to output_dir)
        relative_audio_path = f"Speech WAV Files/{ethnicity} {age_group}/{audio_filename}.wav"
        
        # Get speaker ID (use a simple mapping)
        speaker_id = hash(f"{ethnicity}_{age_group}") % 1000  # Simple hash-based speaker ID
        
        tts_data.append({
            'wav': relative_audio_path,
            'text': transcript,
            'speaker_idx': speaker_id,
        })
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df)} files...")
    
    print(f"\nTotal TTS samples: {len(tts_data)}")
    
    # Create DataFrame
    tts_df = pd.DataFrame(tts_data)
    
    # Split into train/val/test
    train_df, temp_df = train_test_split(
        tts_df,
        test_size=(1 - train_ratio),
        random_state=42,
        shuffle=True
    )
    
    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size),
        random_state=42,
        shuffle=True
    )
    
    print(f"\nSplits:")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(tts_df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/len(tts_df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/len(tts_df)*100:.1f}%)")
    
    # Save CSV files (TTS format: wav|text|speaker_idx)
    train_csv = output_dir / "train.csv"
    val_csv = output_dir / "val.csv"
    test_csv = output_dir / "test.csv"
    
    train_df.to_csv(train_csv, sep='|', index=False, header=False)
    val_df.to_csv(val_csv, sep='|', index=False, header=False)
    test_df.to_csv(test_csv, sep='|', index=False, header=False)
    
    print(f"\nSaved CSV files:")
    print(f"  Train: {train_csv}")
    print(f"  Val:   {val_csv}")
    print(f"  Test:  {test_csv}")
    
    # Copy audio files to output directory (or create symlinks)
    # For now, we'll use the original TIS directory structure
    print(f"\nNote: Audio files are referenced from: {tis_dir}")
    print(f"      Make sure to use --tts_data_dir {tis_dir} when training TTS")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Convert TIS dataset to TTS format")
    parser.add_argument("--tis_dir", type=str, required=True, help="Path to TIS dataset directory")
    parser.add_argument("--output_dir", type=str, default="./tis_tts_data", help="Output directory for TTS CSV files")
    parser.add_argument("--use_asr", action="store_true", help="Use Whisper ASR to extract transcripts")
    parser.add_argument("--placeholder_text", type=str, default=None, help="Placeholder text if not using ASR")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test set ratio")
    
    args = parser.parse_args()
    
    # Validate ratios
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")
    
    create_tts_csv_from_tis(
        tis_dir=args.tis_dir,
        output_dir=args.output_dir,
        use_asr=args.use_asr,
        placeholder_text=args.placeholder_text,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    
    print("\n" + "=" * 80)
    print("TIS to TTS conversion complete!")
    print("=" * 80)
    print("\nNext steps:")
    print(f"1. Train TTS with RL:")
    print(f"   python run_tts_with_rl.py \\")
    print(f"       --tts_data_dir {args.tis_dir} \\")
    print(f"       --train_csv {args.output_dir}/train.csv \\")
    print(f"       --val_csv {args.output_dir}/val.csv \\")
    print(f"       --hubert_checkpoint <path_to_hubert_checkpoint>")


if __name__ == "__main__":
    main()





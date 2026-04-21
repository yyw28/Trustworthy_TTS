#!/usr/bin/env python3
"""
Train TTS model with Reinforcement Learning.

This script trains the Tacotron2 TTS model with RL to optimize GST weights
for trustworthiness using HuBERT classifier rewards.
"""
import argparse
import sys
from pathlib import Path

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from tspeech.model.rl_gst_tts import TTSModel
from tspeech.data.tts.datamodule import TTSDatamodule


def main():
    parser = argparse.ArgumentParser(description="Train TTS with RL")
    
    # Data arguments
    parser.add_argument("--tts_data_dir", type=str, required=True, help="Directory containing CSV files")
    parser.add_argument("--train_csv", type=str, required=True, help="Training CSV file")
    parser.add_argument("--val_csv", type=str, required=True, help="Validation CSV file")
    parser.add_argument("--test_csv", type=str, default="test.csv", help="Test CSV file (optional)")
    
    # Model arguments
    parser.add_argument("--num_chars", type=int, default=39, help="Number of characters")
    parser.add_argument("--encoded_dim", type=int, default=512, help="Encoder dimension")
    parser.add_argument("--encoder_kernel_size", type=int, default=5, help="Encoder kernel size")
    parser.add_argument("--num_mels", type=int, default=80, help="Number of mel bins")
    parser.add_argument("--prenet_dim", type=int, default=256, help="Prenet dimension")
    parser.add_argument("--att_rnn_dim", type=int, default=1024, help="Attention RNN dimension")
    parser.add_argument("--att_dim", type=int, default=128, help="Attention dimension")
    parser.add_argument("--rnn_hidden_dim", type=int, default=1024, help="RNN hidden dimension")
    parser.add_argument("--postnet_dim", type=int, default=512, help="Postnet dimension")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--speaker_tokens_enabled", action="store_true", help="Enable speaker tokens")
    parser.add_argument("--speaker_count", type=int, default=1, help="Number of speakers")
    
    # RL arguments
    parser.add_argument("--use_bert_gst", action="store_true", default=True, help="Use BERT for GST")
    parser.add_argument("--bert_model_name", type=str, default="bert-base-uncased", help="BERT model name")
    parser.add_argument("--hubert_checkpoint", type=str, required=True, help="HuBERT checkpoint path")
    parser.add_argument("--hubert_model_name", type=str, default="facebook/hubert-base-ls960", help="HuBERT model name")
    parser.add_argument("--use_rl_training", action="store_true", default=True, help="Enable RL training")
    parser.add_argument("--rl_temperature", type=float, default=1.0, help="RL sampling temperature")
    parser.add_argument("--rl_entropy_coef", type=float, default=0.01, help="RL entropy coefficient")
    parser.add_argument("--use_vocoder", action="store_true", default=True, help="Use vocoder for RL")
    parser.add_argument("--vocoder_checkpoint_dir", type=str, default=None, help="Vocoder checkpoint directory")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loader workers")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--accelerator", type=str, default="gpu", choices=["gpu", "cpu", "mps"], help="Accelerator")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    parser.add_argument("--precision", type=str, default="16-mixed", help="Precision (16-mixed, 32, etc.)")
    
    # Checkpoint arguments
    parser.add_argument("--tacotron2_checkpoint", type=str, default=None, help="Pre-trained Tacotron2 checkpoint (optional)")
    
    # Audio saving arguments
    parser.add_argument("--save_audio_dir", type=str, default=None, help="Directory to save generated audio")
    parser.add_argument("--save_audio_every_n_steps", type=int, default=100, help="Save audio every N steps")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./lightning_logs", help="Output directory for logs and checkpoints")
    parser.add_argument("--experiment_name", type=str, default="tts_rl", help="Experiment name")
    
    args = parser.parse_args()
    
    # Set float32 matmul precision
    torch.set_float32_matmul_precision("high")
    
    # Initialize data module
    data_module = TTSDatamodule(
        dataset_dir=args.tts_data_dir,
        csv_train=Path(args.train_csv).name,
        csv_val=Path(args.val_csv).name,
        csv_test=args.test_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_mels=args.num_mels,
        sample_rate=22050,  # Standard for TTS
        expand_abbreviations=True,
        allowed_chars="!'(),.:;? \\-abcdefghijklmnopqrstuvwxyz",
        end_token="^",
        silence=0,
        trim=True,
        trim_top_db=40,
        trim_frame_length=1024,
    )
    
    # Initialize model
    model = TTSModel(
        num_chars=args.num_chars,
        encoded_dim=args.encoded_dim,
        encoder_kernel_size=args.encoder_kernel_size,
        num_mels=args.num_mels,
        prenet_dim=args.prenet_dim,
        att_rnn_dim=args.att_rnn_dim,
        att_dim=args.att_dim,
        rnn_hidden_dim=args.rnn_hidden_dim,
        postnet_dim=args.postnet_dim,
        dropout=args.dropout,
        speaker_tokens_enabled=args.speaker_tokens_enabled,
        speaker_count=args.speaker_count,
        use_bert_gst=args.use_bert_gst,
        bert_model_name=args.bert_model_name,
        use_hubert_classifier=True,
        hubert_model_name=args.hubert_model_name,
        hubert_checkpoint_path=args.hubert_checkpoint,
        use_rl_training=args.use_rl_training,
        rl_temperature=args.rl_temperature,
        rl_entropy_coef=args.rl_entropy_coef,
        use_vocoder=args.use_vocoder,
        vocoder_checkpoint_dir=args.vocoder_checkpoint_dir,
        save_audio_dir=args.save_audio_dir,
        save_audio_every_n_steps=args.save_audio_every_n_steps,
    )
    
    # Load pre-trained checkpoint if provided
    if args.tacotron2_checkpoint:
        print(f"Loading pre-trained Tacotron2 checkpoint: {args.tacotron2_checkpoint}")
        checkpoint = torch.load(args.tacotron2_checkpoint, map_location="cpu")
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("✓ Pre-trained checkpoint loaded")
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=Path(args.output_dir) / args.experiment_name,
            filename="checkpoint-{epoch}-{step}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=10,
        ),
    ]
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name=args.experiment_name,
    )
    
    # Initialize trainer
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=callbacks,
        logger=logger,
        precision=args.precision,
        log_every_n_steps=10,
    )
    
    # Train
    print("=" * 80)
    print("Starting TTS RL Training")
    print("=" * 80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"RL training: {args.use_rl_training}")
    print(f"HuBERT checkpoint: {args.hubert_checkpoint}")
    print("=" * 80)
    
    trainer.fit(model, data_module)
    
    print("=" * 80)
    print("Training complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

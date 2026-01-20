#fine-tune TIS
PYTHONPATH=src python3 -m tspeech.htmodel fit --config config/ht-config.json 


#test
PYTHONPATH=src python3 -m tspeech.htmodel test --config config/ht-config.json --ckpt_path lightning_logs/ht-finetune/version_1/checkpoints/checkpoint-epoch=27-validation_f1=0.85496.ckpt
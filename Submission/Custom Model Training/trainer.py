import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from pyabsa import AspectPolarityClassification as APC

dataset_path = "geopolitical"
config = APC.APCConfigManager.get_apc_config_english()
config.model = APC.APCModelList.FAST_LSA_T_V2

config.pretrained_bert = "microsoft/deberta-v3-base"

config.max_seq_len = 128  

config.batch_size = 16    
config.use_amp = True

config.num_epoch = 100            
config.learning_rate = 1e-5  
config.l2reg = 1e-5
config.seed = 42
config.log_step = 10               
config.evaluate_begin = 2          
config.patience = 10               

# Start Training
trainer = APC.APCTrainer(
    config=config,
    dataset=dataset_path,
    checkpoint_save_mode=1,        
    auto_device=True 
)

print("🏎️ GPU-Accelerated Training complete! Check the 'checkpoints' folder.")
import os
import json

# Default configuration settings
DEFAULT_CONFIG = {
    "model": {
        "n_layers": 4,
        "n_heads": 4,
        "d_model": 256,
        "d_ff": 1024,
        "max_seq_len": 512,
        "vocab_size": 5000
    },
    "training": {
        "batch_size": 8,
        "lr": 3e-4,
        "epochs": 5,
        "grad_accum_steps": 4,
        "dropout": 0.1,
        "early_stopping_patience": 3,
        "val_split": 0.1,
        "grad_clip": 1.0,
        "weight_decay": 0.1,
        "train_samples": 1,
        "train_sample_len": 32,
        "train_sample_prompt": ""
    },
    "generation": {
        "max_new_tokens": 256,
        "temperature": 0.8,
        "top_p": 0.9,
        "eos_token": "<EOS>"
    },
    "chat": {
        "user_prefix": "[USER]",
        "gpt_prefix": "[GPT]",
        "human_role": "[HUMAN]",
        "gpt_role": "[GPT]"
    },
    "paths": {
        "data_dir": "data",
        "processed_data_dir": "data/processed",
        "checkpoint_dir": "model/checkpoint",
        "tokenizer_dir": "tokenizer",
        "best_model_path": "model/best/best.pt"
    }
}

# Path handling for config/config.json
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")

def load_config():
    """Load configuration from JSON or create it if missing."""
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)
        
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        return DEFAULT_CONFIG
    
    with open(CONFIG_FILE, "r") as f:
        config_data = json.load(f)
        
    # Shallow merge new defaults into existing config
    updated = False
    for category, settings in DEFAULT_CONFIG.items():
        if category not in config_data:
            config_data[category] = settings
            updated = True
        elif isinstance(settings, dict):
            for k, v in settings.items():
                if k not in config_data[category]:
                    config_data[category][k] = v
                    updated = True
                    
    if updated:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config_data, f, indent=4)
            
    return config_data

# Load the configuration data
_config = load_config()

# Flatten configuration for module-level access (maintains compatibility)
# Model Architecture
n_layers = _config.get("model", {}).get("n_layers", DEFAULT_CONFIG["model"]["n_layers"])
n_heads = _config.get("model", {}).get("n_heads", DEFAULT_CONFIG["model"]["n_heads"])
d_model = _config.get("model", {}).get("d_model", DEFAULT_CONFIG["model"]["d_model"])
d_ff = _config.get("model", {}).get("d_ff", DEFAULT_CONFIG["model"]["d_ff"])
max_seq_len = _config.get("model", {}).get("max_seq_len", DEFAULT_CONFIG["model"]["max_seq_len"])
vocab_size = _config.get("model", {}).get("vocab_size", DEFAULT_CONFIG["model"]["vocab_size"])

# Training Hyperparameters
batch_size = _config.get("training", {}).get("batch_size", DEFAULT_CONFIG["training"]["batch_size"])
lr = _config.get("training", {}).get("lr", DEFAULT_CONFIG["training"]["lr"])
epochs = _config.get("training", {}).get("epochs", DEFAULT_CONFIG["training"]["epochs"])
grad_accum_steps = _config.get("training", {}).get("grad_accum_steps", DEFAULT_CONFIG["training"]["grad_accum_steps"])
dropout = _config.get("training", {}).get("dropout", DEFAULT_CONFIG["training"]["dropout"])
early_stopping_patience = _config.get("training", {}).get("early_stopping_patience", DEFAULT_CONFIG["training"]["early_stopping_patience"])
val_split = _config["training"].get("val_split", 0.1)
grad_clip = _config["training"].get("grad_clip", 1.0)
weight_decay = _config["training"].get("weight_decay", 0.1)
train_samples = _config["training"].get("train_samples", 1)
train_sample_len = _config["training"].get("train_sample_len", 32)
train_sample_prompt = _config["training"].get("train_sample_prompt", "")

# Generation Parameters
max_new_tokens = _config["generation"].get("max_new_tokens", 256)
temperature = _config["generation"].get("temperature", 0.8)
top_p = _config["generation"].get("top_p", 0.9)
eos_token = _config["generation"].get("eos_token", "<EOS>")

# Chat UI Parameters
user_prefix = _config["chat"].get("user_prefix", "[USER]")
gpt_prefix = _config["chat"].get("gpt_prefix", "[GPT]")
human_role = _config["chat"].get("human_role", "[HUMAN]")
gpt_role = _config["chat"].get("gpt_role", "[GPT]")

# Data & Checkpoints
data_dir = os.path.join(BASE_DIR, _config.get("paths", {}).get("data_dir", DEFAULT_CONFIG["paths"]["data_dir"]))
processed_data_dir = os.path.join(BASE_DIR, _config.get("paths", {}).get("processed_data_dir", DEFAULT_CONFIG["paths"]["processed_data_dir"]))
checkpoint_dir = os.path.join(BASE_DIR, _config.get("paths", {}).get("checkpoint_dir", DEFAULT_CONFIG["paths"]["checkpoint_dir"]))
best_model_path = os.path.join(BASE_DIR, _config.get("paths", {}).get("best_model_path", DEFAULT_CONFIG["paths"]["best_model_path"]))
best_model_dir = os.path.dirname(best_model_path)
tokenizer_dir = os.path.join(BASE_DIR, _config.get("paths", {}).get("tokenizer_dir", DEFAULT_CONFIG["paths"]["tokenizer_dir"]))

# Ensure nested paths are correct relative to project root
REQUIRED_DIRS = [data_dir, processed_data_dir, checkpoint_dir, best_model_dir, tokenizer_dir]

def ensure_dirs():
    """Ensure all required directories exist."""
    # Ensure config dir exists (already done in load_config, but for safety)
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)
        
    for d in REQUIRED_DIRS:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
            print(f"Created directory: {d}")

if __name__ == "__main__":
    ensure_dirs()
    print(f"Configuration loaded from {CONFIG_FILE}")

import torch
import configparser
from model import Transformer

print("=========================================")
print("  AXIOM AI MODEL TESTER")
print("=========================================\n")

config = configparser.ConfigParser()
config.read('configs/config.ini')

vocab_size = int(config['MODEL']['vocab_size'])

# ----------------------------------------------------
# TEST 1: Default Dense Feed-Forward Network
# ----------------------------------------------------
config.set('MODEL', 'use_moe', 'False')

print("--- [TEST 1] Standard Dense Network ---")
model = Transformer(config['MODEL'])
print(f"Num Experts: {getattr(model.blocks[0].ffn, 'num_experts', 'MoE Disabled')}")

dummy_input = torch.randint(0, vocab_size, (2, 64))  # Batch=2, Seq=64
logits, aux_loss = model(dummy_input)

print(f"Logits shape: {logits.shape}")
print(f"Aux Loss: {aux_loss}")

loss = logits.sum() + (0.01 * aux_loss if aux_loss != 0.0 else 0)
loss.backward()
print("Backward pass executed successfully! Gradients are flowing.\n")

# ----------------------------------------------------
# TEST 2: Mixture of Experts Network
# ----------------------------------------------------
config.set('MODEL', 'use_moe', 'True')

print("--- [TEST 2] Mixture of Experts (MoE) ---")
model = Transformer(config['MODEL'])
print(f"Num Experts Configured: {model.blocks[0].ffn.num_experts}")

dummy_input = torch.randint(0, vocab_size, (2, 64))  # Batch=2, Seq=64
logits, aux_loss = model(dummy_input)

print(f"Logits shape: {logits.shape}")
print(f"Aux Loss: {aux_loss.item()}")

loss = logits.sum() + (0.01 * aux_loss)
loss.backward()
print("Backward pass executed successfully! Gradients are flowing.\n")

print("All tests passed.")

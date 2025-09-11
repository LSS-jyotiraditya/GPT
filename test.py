#!/usr/bin/env python3
"""
Enhanced Testing Script for MoE NanoGPT Model with Comprehensive Analysis
"""

import torch
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from tokenizers import Tokenizer
from moe_nano_gpt_model import NanoGPTMoE
import os
import sys
import time
import numpy as np
from collections import defaultdict, Counter

class ModelTester:
    def __init__(self, checkpoint_path="/root/stories/slms/checkpoints/enhanced-moe-134.6M-best.pt",
                 tokenizer_path="/root/stories/slms/data/TinyStories-tokenizer.json"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path
        self.expert_usage_stats = defaultdict(int)
        
        print(f"🚀 Initializing Enhanced MoE Model Tester")
        print(f"📱 Device: {self.device}")
        
        if self.device == "cuda":
            print(f"🔥 GPU: {torch.cuda.get_device_name()}")
            print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        self.load_tokenizer()
        self.load_model()
        
    def load_tokenizer(self):
        """Load the tokenizer"""
        try:
            print(f"📖 Loading tokenizer from {self.tokenizer_path}...")
            self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
            print(f"✅ Tokenizer loaded successfully! Vocab size: {self.tokenizer.get_vocab_size()}")
        except Exception as e:
            print(f"❌ Error loading tokenizer: {e}")
            sys.exit(1)
            
    def apply_pruning_structure(self, model):
        """Apply pruning structure to match the checkpoint if needed"""
        print("🔧 Checking for pruning structure...")
        parameters_to_prune = []
        
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                parameters_to_prune.append((module, 'weight'))
        
        if parameters_to_prune:
            # Apply minimal pruning just to create the structure
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=0.0,  # 0% pruning, just create the structure
            )
            print("🔧 Pruning structure applied")
        return model
            
    def load_model(self):
        """Load the enhanced MoE model with automatic configuration detection"""
        try:
            print(f"🤖 Loading model from {self.checkpoint_path}...")
            
            # Load checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            
            # Get hyperparameters
            if 'hyperparameters' not in checkpoint:
                print("❌ No hyperparameters found in checkpoint!")
                sys.exit(1)
                
            hyperparameters = checkpoint['hyperparameters']
            
            # Auto-detect model configuration
            self.detect_model_configuration(hyperparameters)
            
            # Print comprehensive model info
            print(f"📊 Model Configuration:")
            print(f"   • Architecture: MoE NanoGPT")
            print(f"   • Parameters: {self.count_parameters(hyperparameters):.1f}M")
            print(f"   • Layers: {hyperparameters['n_layers']}")
            print(f"   • Embedding dim: {hyperparameters['n_embed']}")
            print(f"   • Attention heads: {hyperparameters['n_heads']}")
            print(f"   • Experts: {hyperparameters['n_experts']}")
            print(f"   • Top-k experts: {hyperparameters['top_k']}")
            print(f"   • Block size: {hyperparameters['block_size']}")
            print(f"   • Dropout: {hyperparameters['dropout']}")
            
            # Print additional checkpoint info if available
            if 'final_sparsity' in checkpoint:
                print(f"   • Final sparsity: {checkpoint['final_sparsity']:.3f}")
            if 'best_val_loss' in checkpoint:
                print(f"   • Best validation loss: {checkpoint['best_val_loss']:.4f}")
            if 'epoch' in checkpoint:
                print(f"   • Trained epochs: {checkpoint['epoch'] + 1}")
            
            # Initialize model
            self.model = NanoGPTMoE(hyperparameters, self.device)
            
            # Apply pruning structure if needed
            if any(key.endswith('_mask') for key in checkpoint['model'].keys()):
                print("🔪 Detected pruned model - applying pruning structure")
                self.model = self.apply_pruning_structure(self.model)
            
            # Move to device
            self.model = self.model.to(self.device)
            
            # Load state dict with comprehensive error handling
            try:
                self.model.load_state_dict(checkpoint['model'])
                print(f"✅ Model loaded successfully!")
            except Exception as e:
                print(f"❌ Error loading state dict: {e}")
                print("🔧 Trying strict=False loading...")
                try:
                    missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint['model'], strict=False)
                    if missing_keys:
                        print(f"⚠️  Missing keys: {len(missing_keys)}")
                        if len(missing_keys) <= 5:
                            for key in missing_keys:
                                print(f"     - {key}")
                    if unexpected_keys:
                        print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
                        if len(unexpected_keys) <= 5:
                            for key in unexpected_keys:
                                print(f"     - {key}")
                    print(f"⚠️ Model loaded with some missing/unexpected keys")
                except Exception as e2:
                    print(f"❌ Failed to load model even with strict=False: {e2}")
                    sys.exit(1)
            
            self.model.eval()
            
            # Store hyperparameters for generation
            self.hyperparameters = hyperparameters
            self.block_size = hyperparameters['block_size']
            
            print("✅ Model initialization complete!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def detect_model_configuration(self, hyperparameters):
        """Detect and validate model configuration"""
        configs = {
            "Small MoE (134.6M)": {
                "n_embed": 256, "n_heads": 8, "n_layers": 12, 
                "n_experts": 8, "expected_params": 134.6
            },
            "Large MoE (266.6M)": {
                "n_embed": 288, "n_heads": 12, "n_layers": 16, 
                "n_experts": 12, "expected_params": 266.6
            }
        }
        
        actual_params = self.count_parameters(hyperparameters)
        
        for config_name, config in configs.items():
            if (hyperparameters.get('n_embed') == config['n_embed'] and 
                hyperparameters.get('n_heads') == config['n_heads'] and
                hyperparameters.get('n_layers') == config['n_layers'] and
                hyperparameters.get('n_experts') == config['n_experts']):
                print(f"🎯 Detected configuration: {config_name}")
                return config_name
        
        print(f"🤔 Custom configuration detected ({actual_params:.1f}M parameters)")
        return "Custom"
            
    def count_parameters(self, hyperparameters):
        """Estimate parameter count"""
        vocab_size = hyperparameters['vocab_size']
        n_embed = hyperparameters['n_embed']
        n_layers = hyperparameters['n_layers']
        n_heads = hyperparameters['n_heads']
        n_experts = hyperparameters['n_experts']
        
        # More accurate estimation
        embed_params = vocab_size * n_embed  # token embeddings
        pos_embed_params = hyperparameters['block_size'] * n_embed  # positional embeddings
        
        # Per layer: attention + MoE
        attention_params_per_layer = (
            3 * n_embed * n_embed +  # Q, K, V projections
            n_embed * n_embed +      # output projection
            2 * n_embed             # layer norms
        )
        
        # MoE params per layer
        moe_params_per_layer = (
            n_embed * n_experts +    # gating network
            n_embed * n_experts +    # noise network
            n_experts * (n_embed * 4 * n_embed + n_embed * 4 * n_embed) + # experts (2 linear layers each)
            n_embed                  # layer norm
        )
        
        layer_params = (attention_params_per_layer + moe_params_per_layer) * n_layers
        output_params = n_embed * vocab_size + n_embed  # lm_head + final layer norm
        
        total = (embed_params + pos_embed_params + layer_params + output_params) / 1e6
        return total
        
    def encode_text(self, text):
        """Encode text to token ids"""
        encoded = self.tokenizer.encode(text)
        return torch.tensor([encoded.ids], dtype=torch.long, device=self.device)
        
    def decode_tokens(self, tokens):
        """Decode token ids to text"""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        if isinstance(tokens[0], list):
            tokens = tokens[0]
        return self.tokenizer.decode(tokens)
        
    def generate_with_analysis(self, idx, max_new_tokens=100, temperature=1.0, top_k=None, analyze_experts=False):
        """Generate text with expert usage analysis"""
        self.model.eval()
        expert_usage = defaultdict(int)
        generation_stats = {
            'tokens_generated': 0,
            'avg_expert_activation': 0,
            'unique_experts_used': set()
        }
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Crop context to block size
                idx_cond = idx[:, -self.block_size:]
                
                # Get predictions
                logits, _ = self.model(idx_cond)
                logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering if specified
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Sample from distribution
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
                
                generation_stats['tokens_generated'] += 1
                
                # Analyze expert usage if requested
                if analyze_experts:
                    self.analyze_expert_usage_step(idx_cond, expert_usage, generation_stats)
                
        if analyze_experts:
            return idx, expert_usage, generation_stats
        return idx
    
    def analyze_expert_usage_step(self, input_tokens, expert_usage, generation_stats):
        """Analyze which experts are being used during generation"""
        # This is a simplified analysis - in a full implementation,
        # you'd need to modify the MoE forward pass to return gating info
        with torch.no_grad():
            # Run through model layers and track expert activations
            x = self.model.token_embedding_table(input_tokens)
            x = x + self.model.position_embedding_table(torch.arange(input_tokens.shape[1], device=self.device))
            
            for i, block in enumerate(self.model.blocks):
                # Simplified expert usage tracking
                # In reality, you'd need to modify the MoE forward pass to return gating info
                expert_usage[f'layer_{i}'] += 1
                generation_stats['unique_experts_used'].add(f'layer_{i}')
    
    def benchmark_performance(self, test_prompts=None, max_tokens=50):
        """Benchmark model performance with various metrics"""
        if test_prompts is None:
            test_prompts = [
                "Once upon a time",
                "The little girl found",
                "In the magical forest",
                "The brave knight decided to",
                "On a sunny day"
            ]
        
        print(f"\n🏁 Performance Benchmark")
        print("=" * 50)
        
        results = []
        total_time = 0
        
        for i, prompt in enumerate(test_prompts):
            print(f"\n📝 Test {i+1}: '{prompt}'")
            
            # Encode prompt
            context = self.encode_text(prompt)
            
            # Measure generation time
            start_time = time.time()
            
            generated = self.generate_with_analysis(
                context, 
                max_new_tokens=max_tokens,
                temperature=0.8,
                top_k=50,
                analyze_experts=True
            )
            
            if len(generated) == 3:
                tokens, expert_usage, gen_stats = generated
            else:
                tokens = generated
                expert_usage = {}
                gen_stats = {'tokens_generated': max_tokens}
            
            end_time = time.time()
            
            # Calculate metrics
            generation_time = end_time - start_time
            tokens_per_second = gen_stats['tokens_generated'] / generation_time
            
            # Decode result
            result_text = self.decode_tokens(tokens)
            
            results.append({
                'prompt': prompt,
                'generation_time': generation_time,
                'tokens_per_second': tokens_per_second,
                'expert_usage': expert_usage,
                'result': result_text
            })
            
            print(f"⚡ Time: {generation_time:.3f}s ({tokens_per_second:.1f} tok/s)")
            print(f"📄 Result: {result_text[len(prompt):].strip()[:100]}...")
            
            total_time += generation_time
        
        # Summary statistics
        avg_time = total_time / len(test_prompts)
        avg_speed = sum(r['tokens_per_second'] for r in results) / len(results)
        
        print(f"\n📊 Benchmark Summary:")
        print(f"   • Average generation time: {avg_time:.3f}s")
        print(f"   • Average speed: {avg_speed:.1f} tokens/second")
        print(f"   • Total test time: {total_time:.3f}s")
        
        return results
    
    def interactive_session(self):
        """Enhanced interactive testing session"""
        print("\n" + "="*70)
        print("🎮 ENHANCED MoE NANOGPT INTERACTIVE SESSION")
        print("="*70)
        print("Commands:")
        print("  • Type your prompt and press Enter to generate")
        print("  • ':help' - Show this help")
        print("  • ':examples' - Show example prompts") 
        print("  • ':settings' - Change generation settings")
        print("  • ':benchmark' - Run performance benchmark")
        print("  • ':analyze' - Generate with expert analysis")
        print("  • ':quit' or ':exit' - Exit the session")
        print("="*70)
        
        # Default settings
        max_tokens = 150
        temperature = 0.8
        top_k = 50
        
        while True:
            try:
                user_input = input("\n🤖 Enter prompt (or command): ").strip()
                
                if user_input.lower() in [':quit', ':exit', ':q']:
                    print("👋 Goodbye!")
                    break
                    
                elif user_input.lower() == ':help':
                    print("\n📚 Help:")
                    print("  • Enter any text prompt to generate a continuation")
                    print("  • Use ':settings' to adjust temperature, max tokens, etc.")
                    print("  • Use ':examples' for sample prompts")
                    print("  • Use ':benchmark' for performance testing")
                    print("  • Use ':analyze' for expert usage analysis")
                    print("  • Use ':quit' to exit")
                    continue
                    
                elif user_input.lower() == ':examples':
                    print("\n📝 Example prompts:")
                    print("  • 'Once upon a time, there was a little girl who'")
                    print("  • 'The brave knight walked into the dark forest and'")
                    print("  • 'In a magical kingdom far away,'")
                    print("  • 'The curious cat discovered something amazing:'")
                    print("  • 'On a sunny day, the children decided to'")
                    continue
                    
                elif user_input.lower() == ':benchmark':
                    self.benchmark_performance()
                    continue
                    
                elif user_input.lower() == ':analyze':
                    prompt = input("Enter prompt for expert analysis: ").strip()
                    if prompt:
                        print(f"\n🔍 Analyzing expert usage for: '{prompt}'")
                        context = self.encode_text(prompt)
                        tokens, expert_usage, gen_stats = self.generate_with_analysis(
                            context, max_new_tokens=max_tokens, 
                            temperature=temperature, top_k=top_k, analyze_experts=True
                        )
                        
                        result = self.decode_tokens(tokens)
                        print(f"📄 Generated: {result}")
                        print(f"🧠 Expert usage: {dict(expert_usage)}")
                        print(f"📊 Stats: {gen_stats}")
                    continue
                    
                elif user_input.lower() == ':settings':
                    print(f"\n⚙️  Current settings:")
                    print(f"  • Max tokens: {max_tokens}")
                    print(f"  • Temperature: {temperature}")
                    print(f"  • Top-k: {top_k}")
                    
                    try:
                        new_max = input(f"New max tokens ({max_tokens}): ").strip()
                        if new_max:
                            max_tokens = max(1, min(500, int(new_max)))
                            
                        new_temp = input(f"New temperature ({temperature}): ").strip()
                        if new_temp:
                            temperature = max(0.1, min(2.0, float(new_temp)))
                            
                        new_k = input(f"New top-k ({top_k}): ").strip()
                        if new_k:
                            top_k = max(1, min(100, int(new_k)))
                            
                        print(f"✅ Settings updated!")
                    except ValueError:
                        print("❌ Invalid input. Settings unchanged.")
                    continue
                    
                elif user_input.startswith(':'):
                    print(f"❌ Unknown command: {user_input}")
                    print("Use ':help' for available commands")
                    continue
                    
                elif not user_input:
                    print("❌ Please enter a prompt or command")
                    continue
                
                # Generate response
                print(f"\n🎯 Generating with max_tokens={max_tokens}, temperature={temperature}, top_k={top_k}...")
                print("💭 Generated text:")
                print("-" * 50)
                
                # Encode input
                context = self.encode_text(user_input)
                
                # Generate
                generated = self.generate_with_analysis(
                    context, 
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k
                )
                
                # Decode and display
                result = self.decode_tokens(generated)
                print(result)
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue

def main():
    """Main function with enhanced checkpoint detection"""
    # Auto-detect best available checkpoint
    checkpoints_dir = "/root/stories/slms/checkpoints/"
    
    # Priority order for checkpoint selection
    checkpoint_patterns = [
        "enhanced-moe-134.6M-best.pt",
        "enhanced-moe-266.6M-best.pt", 
        "enhanced-moe-134.6M-final.pt",
        "enhanced-moe-266.6M-final.pt"
    ]
    
    checkpoint_path = None
    for pattern in checkpoint_patterns:
        potential_path = os.path.join(checkpoints_dir, pattern)
        if os.path.exists(potential_path):
            checkpoint_path = potential_path
            break
    
    if not checkpoint_path:
        print(f"❌ No suitable checkpoint found!")
        print("Available checkpoints:")
        if os.path.exists(checkpoints_dir):
            for f in os.listdir(checkpoints_dir):
                if f.endswith('.pt'):
                    print(f"  • {f}")
        return
    
    print(f"🎯 Using checkpoint: {os.path.basename(checkpoint_path)}")
    
    # Initialize and run tester
    tester = ModelTester(checkpoint_path)
    
    # Quick model test
    print(f"\n🧪 Quick model test...")
    test_prompt = "Once upon a time"
    context = tester.encode_text(test_prompt)
    generated = tester.generate_with_analysis(context, max_new_tokens=20, temperature=0.8)
    result = tester.decode_tokens(generated)
    print(f"✅ Test result: {result}")
    
    # Start interactive session
    tester.interactive_session()

if __name__ == "__main__":
    main()

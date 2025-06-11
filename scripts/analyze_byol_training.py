#!/usr/bin/env python3
"""
Analyze BYOL training results to determine if more epochs are needed
"""

import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

#!/usr/bin/env python3
"""
Analyze BYOL training results to determine if more epochs are needed
"""

import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def find_latest_version():
    """Find the latest training version"""
    versions = glob("experiments/tensorboard/byol_isles2022_mps/version_*")
    if not versions:
        versions = glob("experiments/byol_isles2022_mps/lightning_logs/version_*")
    if not versions:
        print("No training logs found!")
        return None
    return sorted(versions)[-1]

def load_metrics_from_tensorboard(version_dir):
    """Load metrics from TensorBoard event files"""
    event_files = glob(os.path.join(version_dir, "events.out.tfevents.*"))
    if not event_files:
        return None
    
    # Use the first event file found
    event_file = event_files[0]
    print(f"Loading from: {event_file}")
    
    # Load events
    ea = EventAccumulator(event_file)
    ea.Reload()
    
    # Extract available scalar tags
    scalar_tags = ea.Tags()['scalars']
    print(f"Available metrics: {scalar_tags}")
    
    # Create a dictionary to store metrics
    metrics_dict = {}
    
    # Extract train_loss if available
    if 'train_loss' in scalar_tags:
        train_loss_events = ea.Scalars('train_loss')
        steps = [e.step for e in train_loss_events]
        values = [e.value for e in train_loss_events]
        metrics_dict['step'] = steps
        metrics_dict['train_loss'] = values
    
    # Extract other relevant metrics
    for tag in ['val_loss', 'momentum', 'epoch']:
        if tag in scalar_tags:
            events = ea.Scalars(tag)
            metrics_dict[tag] = [e.value for e in events]
    
    # Convert to DataFrame
    if metrics_dict:
        # Ensure all arrays have the same length by taking the minimum
        min_len = min(len(v) for v in metrics_dict.values())
        for k in metrics_dict:
            metrics_dict[k] = metrics_dict[k][:min_len]
        
        return pd.DataFrame(metrics_dict)
    
    return None

def analyze_training():
    """Analyze BYOL training results"""
    print("BYOL Training Analysis")
    print("=" * 50)
    
    # Find latest version
    version_dir = find_latest_version()
    if not version_dir:
        return
    
    print(f"Analyzing: {version_dir}")
    
    # Load metrics
    metrics = load_metrics_from_tensorboard(version_dir)
    if metrics is None:
        print("Could not load metrics. Checking checkpoint files...")
        
        # Alternative: check checkpoint filenames for loss values
        ckpt_files = sorted(glob(os.path.join(version_dir, "checkpoints", "*.ckpt")))
        if ckpt_files:
            print(f"\nFound {len(ckpt_files)} checkpoints")
            # Extract loss from filename if present
            for ckpt in ckpt_files[-5:]:  # Last 5 checkpoints
                print(f"  {os.path.basename(ckpt)}")
        return
    
    # Extract training metrics
    train_loss = metrics[metrics['train_loss'].notna()]['train_loss'].values
    epochs = np.arange(len(train_loss))
    
    print(f"\nTraining Summary:")
    print(f"  Total epochs: {len(train_loss)}")
    print(f"  Initial loss: {train_loss[0]:.4f}")
    print(f"  Final loss: {train_loss[-1]:.4f}")
    print(f"  Loss reduction: {(1 - train_loss[-1]/train_loss[0])*100:.1f}%")
    
    # Analyze last 10 epochs
    if len(train_loss) >= 10:
        last_10_losses = train_loss[-10:]
        loss_std = np.std(last_10_losses)
        loss_trend = np.polyfit(range(10), last_10_losses, 1)[0]
        
        print(f"\nLast 10 epochs analysis:")
        print(f"  Mean loss: {np.mean(last_10_losses):.4f}")
        print(f"  Std deviation: {loss_std:.4f}")
        print(f"  Trend (slope): {loss_trend:.6f}")
        
        # Determine if more training needed
        print(f"\nRecommendation:")
        if loss_std < 0.01 and abs(loss_trend) < 0.001:
            print("  ✅ Training has converged - no need for more epochs")
            print("  Loss is stable with minimal variation")
        elif train_loss[-1] > 1.0:
            print("  ⚠️  Consider more training - loss is still high")
            print("  Recommend: 50-100 more epochs")
        elif abs(loss_trend) > 0.01:
            print("  ⚠️  Loss still decreasing - could benefit from more epochs")
            print("  Recommend: 30-50 more epochs")
        else:
            print("  ✅ Training looks good - marginal benefit from more epochs")
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss')
    
    # Add trend line for last 10 epochs
    if len(train_loss) >= 10:
        z = np.polyfit(epochs[-10:], train_loss[-10:], 1)
        p = np.poly1d(z)
        plt.plot(epochs[-10:], p(epochs[-10:]), 'r--', linewidth=2, label='Trend (last 10)')
    
    plt.xlabel('Epoch')
    plt.ylabel('BYOL Loss')
    plt.title('BYOL Training Progress')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add annotations
    plt.annotate(f'Final: {train_loss[-1]:.3f}', 
                xy=(epochs[-1], train_loss[-1]), 
                xytext=(epochs[-1]-5, train_loss[-1]+0.1),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    plt.savefig('byol_training_analysis.png', dpi=150)
    print(f"\nPlot saved as: byol_training_analysis.png")
    plt.show()
    
    # Additional analysis
    print("\nNext steps:")
    print("1. If converged: Extract encoder for downstream tasks")
    print("2. If not converged: Resume training with:")
    print("   python scripts/train_byol.py --config-name=byol_mps training.epochs=50")
    print("3. Test representations with linear evaluation")

if __name__ == "__main__":
    analyze_training()
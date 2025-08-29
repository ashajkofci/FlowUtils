"""
Example demonstrating Logicle and Hyperlog transforms with real FCS data

This example shows how to use FlowUtils with actual flow cytometry data
from an FCS file. It demonstrates the use of FlowCytometryTools for reading
FCS files and then applying FlowUtils transforms.

Usage:
    python fcs_example.py [path/to/fcs/file]

If no FCS file is provided, will use simulated data similar to real FCS data.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from flowutils import transforms

def load_fcs_data(fcs_path):
    """
    Load FCS data using FlowCytometryTools (if available)
    
    Args:
        fcs_path: Path to FCS file
        
    Returns:
        dict: Dictionary with 'data' array, 'channels' list, and metadata
    """
    try:
        # Try to import FlowCytometryTools  
        from FlowCytometryTools import FCMeasurement
        
        print(f"Loading FCS file: {fcs_path}")
        
        # Load the FCS file
        sample = FCMeasurement(ID='Sample', datafile=fcs_path)
        
        # Get the data as numpy array
        data = sample.data.values
        channels = list(sample.data.columns)
        
        # Extract some metadata
        metadata = {
            'total_events': len(data),
            'channels': channels,
            'channel_count': len(channels)
        }
        
        print(f"✓ Successfully loaded FCS file: {fcs_path}")
        print(f"  Events: {len(data):,}")
        print(f"  Channels: {len(channels)}")
        print(f"  Channel names: {', '.join(channels[:5])}{'...' if len(channels) > 5 else ''}")
        
        return {
            'data': data,
            'channels': channels,
            'metadata': metadata,
            'source': 'fcs_file'
        }
        
    except ImportError:
        print("⚠ FlowCytometryTools not available. Install with:")
        print("  pip install FlowCytometryTools")
        print("  or conda install -c bioconda flowcytometrytools")
        return None
        
    except Exception as e:
        print(f"✗ Error loading FCS file {fcs_path}: {str(e)}")
        return None


def create_simulated_fcs_data():
    """
    Create simulated FCS data that mimics real flow cytometry characteristics.
    
    This generates multi-population flow cytometry data with:
    - Main cell population (positive FL1, moderate FL2) 
    - Debris/background (low FL1, low FL2)
    - Compensation artifacts (negative values from spectral spillover)
    - Autofluorescence population (moderate both channels)
    - Statistical noise and realistic parameter ranges
    
    Designed to match typical 4-color flow cytometry experiments with
    fluorescence channels FL1 (FITC) and FL2 (PE) commonly used for
    cell surface markers.
    
    Returns:
        dict: Dictionary with 'data' array (Nx2) and 'channels' list ['FL1', 'FL2']
    """
    print("📊 Creating simulated multi-population FCS data...")
    print("   Mimicking real flow cytometry with FL1 (FITC) and FL2 (PE) channels")
    
    np.random.seed(42)  # For reproducible results
    n_events = 10000
    
    # Simulate typical flow cytometry channels
    # FSC/SSC are usually positive and roughly log-normal
    FSC_A = np.random.lognormal(mean=np.log(50000), sigma=0.5, size=n_events)
    SSC_A = np.random.lognormal(mean=np.log(20000), sigma=0.6, size=n_events)
    
    # Fluorescence channels can have negative values after compensation
    # Create mixed populations with some negative spillover
    positive_pop = np.random.lognormal(mean=np.log(1000), sigma=1.0, size=n_events//2)
    negative_pop = np.random.normal(loc=-50, scale=30, size=n_events//2)
    FL1_A = np.concatenate([positive_pop, negative_pop])
    np.random.shuffle(FL1_A)
    
    # FL2 with different population structure  
    pop1_size = n_events // 3
    pop2_size = n_events // 3
    pop3_size = n_events - pop1_size - pop2_size  # Ensure exact total
    
    FL2_A = np.concatenate([
        np.random.lognormal(mean=np.log(500), sigma=0.8, size=pop1_size),
        np.random.normal(loc=10, scale=50, size=pop2_size),
        np.random.normal(loc=-20, scale=20, size=pop3_size)
    ])
    np.random.shuffle(FL2_A)
    
    data = np.column_stack([FSC_A, SSC_A, FL1_A, FL2_A])
    channels = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A']
    
    print(f"  Simulated events: {n_events:,}")
    print(f"  Channels: {', '.join(channels)}")
    
    return {'data': data, 'channels': channels}


def analyze_transforms(data, channels, fluorescence_channels=None):
    """
    Apply and compare different transforms on the data
    
    Args:
        data: numpy array of flow cytometry data
        channels: list of channel names
        fluorescence_channels: list of channel indices to transform (default: detect FL channels)
    """
    
    # Auto-detect fluorescence channels if not specified
    if fluorescence_channels is None:
        fluorescence_channels = []
        for i, ch in enumerate(channels):
            if any(fl in ch.upper() for fl in ['FL', 'FITC', 'PE', 'APC', 'BV']):
                fluorescence_channels.append(i)
        
        if not fluorescence_channels:
            # If no FL channels found, assume last 2 channels are fluorescence
            fluorescence_channels = list(range(max(0, len(channels)-2), len(channels)))
    
    print(f"\n🔬 Analyzing transforms on channels: {[channels[i] for i in fluorescence_channels]}")
    
    # Transform parameters optimized for flow cytometry
    T = 262144  # Typical full scale
    M = 4.5     # ~4.5 decades
    W = 0.5     # Linear region width  
    A = 0       # No additional negative decades
    
    print(f"Transform parameters: T={T}, M={M}, W={W}, A={A}")
    
    # Apply transforms
    results = {}
    
    # Original data
    results['original'] = data.copy()
    
    # Logicle transform
    print("  Applying Logicle transform...")
    results['logicle'] = transforms.logicle(data, fluorescence_channels, t=T, m=M, w=W, a=A)
    
    # Hyperlog transform  
    print("  Applying Hyperlog transform...")
    results['hyperlog'] = transforms.hyperlog(data, fluorescence_channels, t=T, m=M, w=W, a=A)
    
    # Linear transform (identity for comparison)
    results['linear'] = data.copy()
    
    # Log10 transform (handle negatives by shifting)
    results['log10'] = data.copy()
    for ch_idx in fluorescence_channels:
        ch_data = data[:, ch_idx]
        # Shift to make all values positive, then log
        min_val = ch_data.min()
        shift = max(0, -min_val + 1)
        results['log10'][:, ch_idx] = np.log10(ch_data + shift)
    
    # Print statistics
    print("\n📈 Transform Statistics:")
    print("-" * 50)
    for transform_name, transform_data in results.items():
        print(f"{transform_name.upper():>10}:")
        for ch_idx in fluorescence_channels:
            ch_name = channels[ch_idx]
            ch_data = transform_data[:, ch_idx]
            print(f"    {ch_name}: min={ch_data.min():8.2f}, max={ch_data.max():8.2f}, "
                  f"mean={ch_data.mean():8.2f}, std={ch_data.std():8.2f}")
    
    return results, fluorescence_channels


def create_comparison_plots(results, channels, fluorescence_channels):
    """
    Create comparison plots showing different transforms
    
    Args:
        results: dict of transformed data
        channels: list of channel names  
        fluorescence_channels: list of channel indices that were transformed
    """
    
    if len(fluorescence_channels) < 2:
        print("⚠ Need at least 2 fluorescence channels for FL1-FL2 comparison plot")
        return
    
    # Use first two fluorescence channels for comparison
    ch1_idx, ch2_idx = fluorescence_channels[0], fluorescence_channels[1]
    ch1_name, ch2_name = channels[ch1_idx], channels[ch2_idx]
    
    print(f"\n📊 Creating FL1-FL2 comparison plots: {ch1_name} vs {ch2_name}")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    transforms_to_plot = ['linear', 'log10', 'hyperlog', 'logicle']
    
    for i, transform_name in enumerate(transforms_to_plot):
        ax = axes[i]
        
        if transform_name not in results:
            ax.text(0.5, 0.5, f'{transform_name.title()}\nNot Available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{transform_name.title()} Transform')
            continue
        
        # Get transformed data
        data_transformed = results[transform_name]
        x_data = data_transformed[:, ch1_idx]
        y_data = data_transformed[:, ch2_idx]
        
        # Create density scatter plot
        ax.scatter(x_data, y_data, alpha=0.1, s=1, c='blue', rasterized=True)
        
        ax.set_xlabel(f'{ch1_name} ({transform_name})')
        ax.set_ylabel(f'{ch2_name} ({transform_name})')
        ax.set_title(f'{transform_name.title()} Transform')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Events: {len(x_data):,}\n'
        stats_text += f'X range: [{x_data.min():.1f}, {x_data.max():.1f}]\n'
        stats_text += f'Y range: [{y_data.min():.1f}, {y_data.max():.1f}]'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('fcs_transforms_comparison.png', dpi=150, bbox_inches='tight')
    print("💾 Saved comparison plot: fcs_transforms_comparison.png")
    
    # Show plot if in interactive environment
    try:
        plt.show()
    except:
        pass


def main():
    """Main function"""
    print("🧪 FlowUtils FCS Transform Example")
    print("=" * 50)
    
    # Check if FCS file path provided
    fcs_path = None
    if len(sys.argv) > 1:
        fcs_path = sys.argv[1]
        print(f"📁 Using FCS file: {fcs_path}")
    
    # Load data
    if fcs_path:
        fcs_data = load_fcs_data(fcs_path)
    else:
        fcs_data = None
    
    # Fall back to simulated data if needed
    if fcs_data is None:
        fcs_data = create_simulated_fcs_data()
    
    data = fcs_data['data']
    channels = fcs_data['channels']
    
    print(f"\n📊 Data Summary:")
    print(f"  Shape: {data.shape}")
    print(f"  Events: {data.shape[0]:,}")
    print(f"  Channels: {data.shape[1]}")
    
    # Show data ranges
    print(f"\n📈 Channel Ranges:")
    for i, ch in enumerate(channels):
        ch_data = data[:, i]
        print(f"  {ch}: [{ch_data.min():.1f}, {ch_data.max():.1f}] "
              f"(mean: {ch_data.mean():.1f}, negatives: {np.sum(ch_data < 0)}/{len(ch_data)})")
    
    # Analyze transforms
    results, fluorescence_channels = analyze_transforms(data, channels)
    
    # Create visualization
    try:
        create_comparison_plots(results, channels, fluorescence_channels)
    except Exception as e:
        print(f"⚠ Could not create plots: {e}")
    
    # Test round-trip accuracy for Logicle
    print(f"\n🎯 Round-trip Accuracy Test:")
    print("-" * 30)
    logicle_data = results['logicle']
    
    T, M, W, A = 262144, 4.5, 0.5, 0
    for ch_idx in fluorescence_channels:
        ch_name = channels[ch_idx]
        original_ch = data[:, ch_idx]
        logicle_ch = logicle_data[:, ch_idx]
        
        # Test inverse transform - work on single channel data
        recovered_data = data.copy()
        recovered_data[:, ch_idx] = logicle_ch
        recovered = transforms.logicle_inverse(recovered_data, [ch_idx], t=T, m=M, w=W, a=A)
        error = np.max(np.abs(original_ch - recovered[:, ch_idx]))
        
        print(f"  {ch_name}: max error = {error:.2e}")
    
    print(f"\n✅ Analysis completed successfully!")
    print(f"💡 To use with your own FCS file:")
    print(f"   python {sys.argv[0]} path/to/your/file.fcs")


if __name__ == "__main__":
    main()
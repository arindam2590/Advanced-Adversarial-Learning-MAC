import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_combined_training_loss_plot(baseline_file, curriculum_file, save_path=None):
    """
    Create a combined plot showing training losses/errors for all agents from both models.
    
    Parameters:
    baseline_file (str): Path to baseline DQN training loss CSV/Excel file
    curriculum_file (str): Path to curriculum DQN training loss CSV/Excel file
    save_path (str, optional): Path to save the plot image
    """
    
    if baseline_file.endswith('.xlsx') or baseline_file.endswith('.xls'):
        baseline_data = pd.read_excel(baseline_file)
    else:
        baseline_data = pd.read_csv(baseline_file)
        
    if curriculum_file.endswith('.xlsx') or curriculum_file.endswith('.xls'):
        curriculum_data = pd.read_excel(curriculum_file)
    else:
        curriculum_data = pd.read_csv(curriculum_file)

    episodes = baseline_data.index if 'Episode' not in baseline_data.columns else baseline_data['Episode']
    

    baseline_agent0_loss = baseline_data['Training Error Agent 0']
    baseline_agent1_loss = baseline_data['Training Error Agent 1']
    curriculum_agent0_loss = curriculum_data['Training Error Agent 0']
    curriculum_agent1_loss = curriculum_data['Training Error Agent 1']

    def moving_average(data, window=50):
        return data.rolling(window=window, min_periods=1).mean()
    
    baseline_agent0_ma = moving_average(baseline_agent0_loss)
    baseline_agent1_ma = moving_average(baseline_agent1_loss)
    curriculum_agent0_ma = moving_average(curriculum_agent0_loss)
    curriculum_agent1_ma = moving_average(curriculum_agent1_loss)

    plt.figure(figsize=(14, 8))

    colors = {
        'baseline_0': '#3498db',     
        'baseline_1': '#e74c3c',  
        'curriculum_0': '#2ecc71',   
        'curriculum_1': '#f39c12'    
    }

    line_styles = {
        'baseline': '-',     
        'curriculum': '-'  
    }
    

    plt.plot(episodes, baseline_agent0_loss, color=colors['baseline_0'], alpha=0.2, linewidth=0.5)
    plt.plot(episodes, baseline_agent1_loss, color=colors['baseline_1'], alpha=0.2, linewidth=0.5)
    plt.plot(episodes, curriculum_agent0_loss, color=colors['curriculum_0'], alpha=0.2, linewidth=0.5)
    plt.plot(episodes, curriculum_agent1_loss, color=colors['curriculum_1'], alpha=0.2, linewidth=0.5)
    

    plt.plot(episodes, baseline_agent0_ma, color=colors['baseline_0'], linewidth=2.5, 
             label='Baseline DQN Agent 0', linestyle=line_styles['baseline'])
    plt.plot(episodes, baseline_agent1_ma, color=colors['baseline_1'], linewidth=2.5, 
             label='Baseline DQN Agent 1', linestyle=line_styles['baseline'])
    plt.plot(episodes, curriculum_agent0_ma, color=colors['curriculum_0'], linewidth=2.5, 
             label='Curriculum DQN Agent 0', linestyle=line_styles['curriculum'])
    plt.plot(episodes, curriculum_agent1_ma, color=colors['curriculum_1'], linewidth=2.5, 
             label='Curriculum DQN Agent 1', linestyle=line_styles['curriculum'])
    

    plt.xlabel('Episodes', fontsize=12, fontweight='bold')
    plt.ylabel('Training Loss/Error', fontsize=12, fontweight='bold')
    plt.title('DQN Training Loss Comparison: Baseline vs Curriculum Learning', 
              fontsize=16, fontweight='bold', pad=20)
    

    plt.grid(True, alpha=0.3, linestyle='--', which='both')
    

    plt.legend(fontsize=11, loc='upper right', frameon=True, 
               fancybox=True, shadow=True, framealpha=0.95,
               bbox_to_anchor=(0.98, 0.98))

    plt.xlim(0, len(episodes))

    plt.yscale('log')

    plt.tight_layout()

    plt.minorticks_on()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Plot saved to: {save_path}")

    plt.show()

    print("\n=== Training Loss Statistics ===")
    print(f"Baseline Agent 0 - Final Loss: {baseline_agent0_loss.iloc[-1]:.2e}")
    print(f"Baseline Agent 1 - Final Loss: {baseline_agent1_loss.iloc[-1]:.2e}")
    print(f"Curriculum Agent 0 - Final Loss: {curriculum_agent0_loss.iloc[-1]:.2e}")
    print(f"Curriculum Agent 1 - Final Loss: {curriculum_agent1_loss.iloc[-1]:.2e}")
    print(f"\nAverage Final Loss:")
    print(f"Baseline DQN: {(baseline_agent0_loss.iloc[-1] + baseline_agent1_loss.iloc[-1])/2:.2e}")
    print(f"Curriculum DQN: {(curriculum_agent0_loss.iloc[-1] + curriculum_agent1_loss.iloc[-1])/2:.2e}")

def create_combined_training_loss_plot_from_dataframes(baseline_data, curriculum_data, save_path=None):
    """
    Alternative function if you want to pass DataFrames directly instead of file paths
    """
    episodes = baseline_data.index if 'Episode' not in baseline_data.columns else baseline_data['Episode']

    baseline_agent0_loss = baseline_data['Training Error Agent 0']
    baseline_agent1_loss = baseline_data['Training Error Agent 1']
    curriculum_agent0_loss = curriculum_data['Training Error Agent 0']
    curriculum_agent1_loss = curriculum_data['Training Error Agent 1']

    def moving_average(data, window=50):
        return data.rolling(window=window, min_periods=1).mean()
    
    baseline_agent0_ma = moving_average(baseline_agent0_loss)
    baseline_agent1_ma = moving_average(baseline_agent1_loss)
    curriculum_agent0_ma = moving_average(curriculum_agent0_loss)
    curriculum_agent1_ma = moving_average(curriculum_agent1_loss)

    plt.figure(figsize=(14, 8))
    
    colors = {
        'baseline_0': '#9b59b6',    
        'baseline_1': '#f39c12',
        'curriculum_0': '#9b59b6',
        'curriculum_1': '#f39c12'    
    }

    line_styles = {
        'baseline': '-',     
        'curriculum': '--'   
    }
    
    plt.plot(episodes, baseline_agent0_loss, color=colors['baseline_0'], alpha=0.2, linewidth=0.5)
    plt.plot(episodes, baseline_agent1_loss, color=colors['baseline_1'], alpha=0.2, linewidth=0.5)
    plt.plot(episodes, curriculum_agent0_loss, color=colors['curriculum_0'], alpha=0.2, linewidth=0.5)
    plt.plot(episodes, curriculum_agent1_loss, color=colors['curriculum_1'], alpha=0.2, linewidth=0.5)
    
    plt.plot(episodes, baseline_agent0_ma, color=colors['baseline_0'], linewidth=2.5, 
             label='Baseline DQN Agent 0', linestyle=line_styles['baseline'])
    plt.plot(episodes, baseline_agent1_ma, color=colors['baseline_1'], linewidth=2.5, 
             label='Baseline DQN Agent 1', linestyle=line_styles['baseline'])
    plt.plot(episodes, curriculum_agent0_ma, color=colors['curriculum_0'], linewidth=2.5, 
             label='Curriculum DQN Agent 0', linestyle=line_styles['curriculum'])
    plt.plot(episodes, curriculum_agent1_ma, color=colors['curriculum_1'], linewidth=2.5, 
             label='Curriculum DQN Agent 1', linestyle=line_styles['curriculum'])
    
    plt.xlabel('Episodes', fontsize=12, fontweight='bold')
    plt.ylabel('Training Loss/Error', fontsize=12, fontweight='bold')
    plt.title('DQN Training Loss Comparison: Baseline vs Curriculum Learning', 
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--', which='both')
    plt.legend(fontsize=11, loc='upper right', frameon=True, 
               fancybox=True, shadow=True, framealpha=0.95,
               bbox_to_anchor=(0.98, 0.98))
    plt.xlim(0, len(episodes))
    plt.yscale('log')
    plt.tight_layout()
    plt.minorticks_on()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    baseline_file = "DQN_1500_training_data.csv" 
    curriculum_file = "DQN_1500_training_data_CL_F.csv" 
    
    create_combined_training_loss_plot(baseline_file, curriculum_file, 
                                     save_path="combined_dqn_training_losses.png")


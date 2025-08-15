import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_combined_dqn_plot(baseline_file, curriculum_file, save_path=None):
    """
    Create a combined plot showing cumulative returns for all agents from both models.
    
    Parameters:
    baseline_file (str): Path to baseline DQN results CSV file
    curriculum_file (str): Path to curriculum DQN results CSV file
    save_path (str, optional): Path to save the plot image
    """

    baseline_data = pd.read_csv(baseline_file)
    curriculum_data = pd.read_csv(curriculum_file)

    episodes = baseline_data.index

    baseline_agent0 = baseline_data['Reward Agent 0']
    baseline_agent1 = baseline_data['Reward Agent 1']
    curriculum_agent0 = curriculum_data['Reward Agent 0']
    curriculum_agent1 = curriculum_data['Reward Agent 1']

    def moving_average(data, window=50):
        return data.rolling(window=window, min_periods=1).mean()
    
    baseline_agent0_ma = moving_average(baseline_agent0)
    baseline_agent1_ma = moving_average(baseline_agent1)
    curriculum_agent0_ma = moving_average(curriculum_agent0)
    curriculum_agent1_ma = moving_average(curriculum_agent1)

    plt.figure(figsize=(14, 8))

    colors = {
        'baseline_0': '#3498db',    
        'baseline_1': '#e74c3c',      
        'curriculum_0': '#2ecc71',   
        'curriculum_1': '#f39c12'   
    }
    

    plt.plot(episodes, baseline_agent0, color=colors['baseline_0'], alpha=0.3, linewidth=0.5)
    plt.plot(episodes, baseline_agent1, color=colors['baseline_1'], alpha=0.3, linewidth=0.5)
    plt.plot(episodes, curriculum_agent0, color=colors['curriculum_0'], alpha=0.3, linewidth=0.5)
    plt.plot(episodes, curriculum_agent1, color=colors['curriculum_1'], alpha=0.3, linewidth=0.5)

    plt.plot(episodes, baseline_agent0_ma, color=colors['baseline_0'], linewidth=2, 
             label='Baseline Agent 0', linestyle='-')
    plt.plot(episodes, baseline_agent1_ma, color=colors['baseline_1'], linewidth=2, 
             label='Baseline Agent 1', linestyle='-')
    plt.plot(episodes, curriculum_agent0_ma, color=colors['curriculum_0'], linewidth=2, 
             label='Curriculum DQN Agent 0', linestyle='-')
    plt.plot(episodes, curriculum_agent1_ma, color=colors['curriculum_1'], linewidth=2, 
             label='Curriculum DQN Agent 1', linestyle='-')

    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Cumulative Reward', fontsize=12)
    plt.title('DQN Agent Performance Comparison: Baseline vs Curriculum Learning', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='best')

    plt.xlim(0, len(episodes))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()

if __name__ == "__main__":

    baseline_file = "DQN_1500_training_data.csv"  
    curriculum_file = "DQN_1500_training_data_CL.csv"  

    create_combined_dqn_plot(baseline_file, curriculum_file, save_path="combined_dqn_performance.png")
    


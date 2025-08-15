import argparse
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle


def setup_parser():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-d', '--dqn',
                        action='store_true',
                        help='Simulate the environment with DQN Model: use argument -d or --dqn')
    parser.add_argument('-q', '--doubledqn',
                        action='store_true',
                        help='Simulate the environment with Double DQN Model: use argument -q or --ddqn')
    parser.add_argument('-u', '--dueldqn',
                        action='store_true',
                        help='Simulate the environment with Dueling DQN Model: use argument -u or --dueldqn')
    args = parser.parse_args()
    return args
    
class DataVisualization:
    def __init__(self, episodes, result, model, train_data_filename):
        param_dir = 'Simulation/Utils/'
        with open(param_dir + 'config.json', 'r') as file:
            self.params = json.load(file)
        self.model = model
        self.fig_dir = self.params['DATA_DIR']
        if not os.path.exists(self.fig_dir):
            os.makedirs(self.fig_dir)
        self.train_data_filename = train_data_filename
        self.n_episodes = episodes
        self.returns = result[0]
        self.epsilon_decay_history = result[1]
        self.training_error = result[2]
        self.steps = result[3]
        

        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def save_data(self):
        filepath = self.fig_dir + self.train_data_filename
        df = pd.DataFrame({
            'Reward Agent 0': np.array(self.returns)[:, 0],
            'Reward Agent 1': np.array(self.returns)[:, 1],
            'Steps': self.steps,
            'Epsilon Decay': self.epsilon_decay_history,
            'Training Error Agent 0': np.array(self.training_error)[:, 0],
            'Training Error Agent 1': np.array(self.training_error)[:, 1]
        })
    
        if not os.path.isfile(filepath):
            with pd.ExcelWriter(filepath, mode='w', engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=self.model)
        else:
            with pd.ExcelWriter(filepath, mode='a', if_sheet_exists='replace', engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=self.model)

    def plot_returns(self):
        """Plot episode returns for both agents with moving average"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        episodes = np.arange(1, self.n_episodes + 1)
        agent_0_returns = np.array(self.returns)[:, 0]
        agent_1_returns = np.array(self.returns)[:, 1]
        
        window_size = min(50, self.n_episodes // 10)
        if window_size > 1:
            ma_agent_0 = pd.Series(agent_0_returns).rolling(window=window_size).mean()
            ma_agent_1 = pd.Series(agent_1_returns).rolling(window=window_size).mean()
        else:
            ma_agent_0 = agent_0_returns
            ma_agent_1 = agent_1_returns
        
        ax1.plot(episodes, agent_0_returns, alpha=0.3, color='blue', linewidth=0.5, label='Raw Returns')
        ax1.plot(episodes, ma_agent_0, color='darkblue', linewidth=2, label=f'Moving Average (window={window_size})')
        ax1.set_title(f'{self.model} - Agent 0 Returns Over Episodes', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Cumulative Return')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(episodes, agent_1_returns, alpha=0.3, color='red', linewidth=0.5, label='Raw Returns')
        ax2.plot(episodes, ma_agent_1, color='darkred', linewidth=2, label=f'Moving Average (window={window_size})')
        ax2.set_title(f'{self.model} - Agent 1 Returns Over Episodes', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Cumulative Return')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.fig_dir}{self.model}_returns_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f'Info: Returns plot saved as {self.model}_returns_plot.png')

    def plot_episode_length(self):
        """Plot episode lengths with statistics"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        episodes = np.arange(1, self.n_episodes + 1)
        
        window_size = min(50, self.n_episodes // 10)
        if window_size > 1:
            ma_steps = pd.Series(self.steps).rolling(window=window_size).mean()
        else:
            ma_steps = self.steps
        
        ax1.plot(episodes, self.steps, alpha=0.4, color='green', linewidth=0.8, label='Episode Length')
        ax1.plot(episodes, ma_steps, color='darkgreen', linewidth=2, label=f'Moving Average (window={window_size})')
        ax1.axhline(y=np.mean(self.steps), color='orange', linestyle='--', linewidth=2, label=f'Mean: {np.mean(self.steps):.1f}')
        ax1.set_title(f'{self.model} - Episode Length Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Steps per Episode')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.hist(self.steps, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(x=np.mean(self.steps), color='orange', linestyle='--', linewidth=2, label=f'Mean: {np.mean(self.steps):.1f}')
        ax2.axvline(x=np.median(self.steps), color='red', linestyle='--', linewidth=2, label=f'Median: {np.median(self.steps):.1f}')
        ax2.set_title(f'{self.model} - Distribution of Episode Lengths', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Steps per Episode')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.fig_dir}{self.model}_episode_length_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f'Info: Episode length plot saved as {self.model}_episode_length_plot.png')

    def plot_training_error(self):
        """Plot training error (loss) for both agents"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        episodes = np.arange(1, self.n_episodes + 1)
        agent_0_error = np.array(self.training_error)[:, 0]
        agent_1_error = np.array(self.training_error)[:, 1]
        
        window_size = min(50, self.n_episodes // 10)
        if window_size > 1:
            ma_error_0 = pd.Series(agent_0_error).rolling(window=window_size).mean()
            ma_error_1 = pd.Series(agent_1_error).rolling(window=window_size).mean()
        else:
            ma_error_0 = agent_0_error
            ma_error_1 = agent_1_error
        
        ax1.plot(episodes, agent_0_error, alpha=0.3, color='purple', linewidth=0.5, label='Raw Loss')
        ax1.plot(episodes, ma_error_0, color='darkmagenta', linewidth=2, label=f'Moving Average (window={window_size})')
        ax1.set_title(f'{self.model} - Agent 0 Training Loss Over Episodes', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Training Loss')
        ax1.set_yscale('log')  
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(episodes, agent_1_error, alpha=0.3, color='orange', linewidth=0.5, label='Raw Loss')
        ax2.plot(episodes, ma_error_1, color='darkorange', linewidth=2, label=f'Moving Average (window={window_size})')
        ax2.set_title(f'{self.model} - Agent 1 Training Loss Over Episodes', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Training Loss')
        ax2.set_yscale('log')  
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        ax3.plot(episodes, ma_error_0, color='darkmagenta', linewidth=2, label='Agent 0 (Moving Avg)')
        ax3.plot(episodes, ma_error_1, color='darkorange', linewidth=2, label='Agent 1 (Moving Avg)')
        ax3.set_title(f'{self.model} - Training Loss Comparison', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Episodes')
        ax3.set_ylabel('Training Loss')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.fig_dir}{self.model}_training_error_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f'Info: Training error plot saved as {self.model}_training_error_plot.png')

    def plot_epsilon_decay(self):
        """Plot epsilon decay over episodes"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        episodes = np.arange(1, self.n_episodes + 1)
        
        ax1.plot(episodes, self.epsilon_decay_history, color='teal', linewidth=2, marker='o', markersize=2)
        ax1.set_title(f'{self.model} - Epsilon Decay Over Episodes', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Epsilon Value')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.05)
        
        min_epsilon = np.min(self.epsilon_decay_history)
        max_epsilon = np.max(self.epsilon_decay_history)
        ax1.annotate(f'Max ε: {max_epsilon:.3f}', 
                    xy=(1, max_epsilon), xytext=(10, max_epsilon + 0.1),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
        ax1.annotate(f'Min ε: {min_epsilon:.3f}', 
                    xy=(self.n_episodes, min_epsilon), xytext=(self.n_episodes - 50, min_epsilon + 0.1),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
        
        exploration_ratio = np.sum(self.epsilon_decay_history > 0.1) / len(self.epsilon_decay_history)
        ax2.bar(['Exploration Phase', 'Exploitation Phase'], 
                [exploration_ratio * 100, (1 - exploration_ratio) * 100],
                color=['skyblue', 'lightcoral'], alpha=0.7, edgecolor='black')
        ax2.set_title(f'{self.model} - Exploration vs Exploitation Balance', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Percentage of Episodes (%)')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate([exploration_ratio * 100, (1 - exploration_ratio) * 100]):
            ax2.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.fig_dir}{self.model}_epsilon_decay_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f'Info: Epsilon decay plot saved as {self.model}_epsilon_decay_plot.png')

    def plot_comprehensive_summary(self):
        """Create a comprehensive summary plot with all metrics"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        episodes = np.arange(1, self.n_episodes + 1)
        agent_0_returns = np.array(self.returns)[:, 0]
        agent_1_returns = np.array(self.returns)[:, 1]
        agent_0_error = np.array(self.training_error)[:, 0]
        agent_1_error = np.array(self.training_error)[:, 1]
        
        window_size = min(50, self.n_episodes // 10)
        if window_size > 1:
            ma_agent_0_returns = pd.Series(agent_0_returns).rolling(window=window_size).mean()
            ma_agent_1_returns = pd.Series(agent_1_returns).rolling(window=window_size).mean()
            ma_steps = pd.Series(self.steps).rolling(window=window_size).mean()
        else:
            ma_agent_0_returns = agent_0_returns
            ma_agent_1_returns = agent_1_returns
            ma_steps = self.steps
        
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(episodes, ma_agent_0_returns, color='blue', linewidth=2, label='Agent 0')
        ax1.plot(episodes, ma_agent_1_returns, color='red', linewidth=2, label='Agent 1')
        ax1.set_title(f'{self.model} - Returns Comparison', fontweight='bold')
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(episodes, self.epsilon_decay_history, color='teal', linewidth=2)
        ax2.set_title('Epsilon Decay', fontweight='bold')
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Epsilon')
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(episodes, ma_steps, color='green', linewidth=2)
        ax3.set_title('Episode Length', fontweight='bold')
        ax3.set_xlabel('Episodes')
        ax3.set_ylabel('Steps')
        ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(gs[1, 1:])
        ax4.semilogy(episodes, agent_0_error, alpha=0.3, color='purple', linewidth=0.5)
        ax4.semilogy(episodes, agent_1_error, alpha=0.3, color='orange', linewidth=0.5)
        if window_size > 1:
            ma_error_0 = pd.Series(agent_0_error).rolling(window=window_size).mean()
            ma_error_1 = pd.Series(agent_1_error).rolling(window=window_size).mean()
            ax4.semilogy(episodes, ma_error_0, color='darkmagenta', linewidth=2, label='Agent 0')
            ax4.semilogy(episodes, ma_error_1, color='darkorange', linewidth=2, label='Agent 1')
        ax4.set_title('Training Loss Comparison', fontweight='bold')
        ax4.set_xlabel('Episodes')
        ax4.set_ylabel('Loss (log scale)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        ax5 = fig.add_subplot(gs[2, :])
        stats_data = {
            'Metric': ['Avg Return Agent 0', 'Avg Return Agent 1', 'Avg Episode Length', 
                      'Final Epsilon', 'Max Return Agent 0', 'Max Return Agent 1'],
            'Value': [np.mean(agent_0_returns), np.mean(agent_1_returns), np.mean(self.steps),
                     self.epsilon_decay_history[-1], np.max(agent_0_returns), np.max(agent_1_returns)]
        }
        
        ax5.axis('tight')
        ax5.axis('off')
        table = ax5.table(cellText=[[metric, f'{value:.3f}'] for metric, value in zip(stats_data['Metric'], stats_data['Value'])],
                         colLabels=['Performance Metric', 'Value'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax5.set_title('Training Summary Statistics', fontweight='bold', pad=20)
        
        plt.suptitle(f'{self.model} - Comprehensive Training Analysis', fontsize=16, fontweight='bold')
        plt.savefig(f'{self.fig_dir}{self.model}_comprehensive_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f'Info: Comprehensive summary plot saved as {self.model}_comprehensive_summary.png')
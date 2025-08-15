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


    def plot_performance_metrics(self, excel_filename=None):
        """Plot comprehensive performance metrics from Excel data"""
        if excel_filename is None:
            import glob
            pattern = os.path.join(self.fig_dir, "combat_performance_metrics_*.xlsx")
            files = glob.glob(pattern)
            if not files:
                print("No performance metrics Excel file found")
                return
            excel_filename = max(files, key=os.path.getctime)
        
        try:
            df = pd.read_excel(excel_filename, sheet_name='Performance_Data')
            
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
            
            ax1 = fig.add_subplot(gs[0, :2])
            episodes = df['episode']
            cps = df['combat_performance_score']
            
            ax1.plot(episodes, cps, alpha=0.3, color='purple', linewidth=0.8, label='Raw CPS')
            window_size = min(50, len(df) // 10)
            if window_size > 1:
                ma_cps = pd.Series(cps).rolling(window=window_size).mean()
                ax1.plot(episodes, ma_cps, color='darkmagenta', linewidth=3, 
                        label=f'Moving Average (window={window_size})')
            
            ax1.set_title('Combat Performance Score Over Time', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Episodes')
            ax1.set_ylabel('CPS')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_ylim(0, 1)
            
            ax2 = fig.add_subplot(gs[0, 2:])
            success_rate = df['success_rate']
            ax2.plot(episodes, success_rate, alpha=0.4, color='green', linewidth=0.8, label='Success Rate')
            if window_size > 1:
                ma_success = pd.Series(success_rate).rolling(window=window_size).mean()
                ax2.plot(episodes, ma_success, color='darkgreen', linewidth=3, 
                        label=f'Moving Average (window={window_size})')
            
            ax2.set_title('Success Rate Over Time', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Episodes')
            ax2.set_ylabel('Success Rate')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.set_ylim(0, 1)
            
            ax3 = fig.add_subplot(gs[1, 0])
            time_to_capture = df['time_to_capture']
            ax3.plot(episodes, time_to_capture, alpha=0.4, color='orange', linewidth=0.8)
            if window_size > 1:
                ma_capture = pd.Series(time_to_capture).rolling(window=window_size).mean()
                ax3.plot(episodes, ma_capture, color='darkorange', linewidth=3, 
                        label=f'MA (w={window_size})')
            
            ax3.set_title('Time to Capture', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Episodes')
            ax3.set_ylabel('Steps')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            ax4 = fig.add_subplot(gs[1, 1])
            deception_resistance = df['deception_resistance_score']
            ax4.plot(episodes, deception_resistance, alpha=0.4, color='red', linewidth=0.8)
            if window_size > 1:
                ma_deception = pd.Series(deception_resistance).rolling(window=window_size).mean()
                ax4.plot(episodes, ma_deception, color='darkred', linewidth=3, 
                        label=f'MA (w={window_size})')
            
            ax4.set_title('Deception Resistance', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Episodes')
            ax4.set_ylabel('Resistance Score')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            ax4.set_ylim(0, 1)
            
            ax5 = fig.add_subplot(gs[1, 2:])
            agent0_returns = df['episode_return_agent0']
            agent1_returns = df['episode_return_agent1']
            
            ax5.plot(episodes, agent0_returns, alpha=0.4, color='blue', linewidth=0.8, label='Agent 0')
            ax5.plot(episodes, agent1_returns, alpha=0.4, color='red', linewidth=0.8, label='Agent 1')
            
            if window_size > 1:
                ma_agent0 = pd.Series(agent0_returns).rolling(window=window_size).mean()
                ma_agent1 = pd.Series(agent1_returns).rolling(window=window_size).mean()
                ax5.plot(episodes, ma_agent0, color='darkblue', linewidth=3, label='Agent 0 MA')
                ax5.plot(episodes, ma_agent1, color='darkred', linewidth=3, label='Agent 1 MA')
            
            ax5.set_title('Episode Returns Comparison', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Episodes')
            ax5.set_ylabel('Episode Return')
            ax5.grid(True, alpha=0.3)
            ax5.legend()
            
            if 'curriculum_level' in df.columns:
                ax6 = fig.add_subplot(gs[2, :2])
                curriculum_levels = df['curriculum_level']
                ax6.plot(episodes, curriculum_levels, color='teal', linewidth=2, marker='o', markersize=3)
                ax6.set_title('Curriculum Level Progression', fontsize=14, fontweight='bold')
                ax6.set_xlabel('Episodes')
                ax6.set_ylabel('Curriculum Level')
                ax6.grid(True, alpha=0.3)
                ax6.set_ylim(0.5, max(curriculum_levels) + 0.5)
            
            if 'curriculum_level' in df.columns:
                ax7 = fig.add_subplot(gs[2, 2:])
                curriculum_data = []
                levels = sorted(df['curriculum_level'].unique())
                
                for level in levels:
                    level_data = df[df['curriculum_level'] == level]
                    curriculum_data.append(level_data['combat_performance_score'].values)
                
                ax7.boxplot(curriculum_data, labels=[f'L{int(l)}' for l in levels])
                ax7.set_title('CPS Distribution by Curriculum Level', fontsize=12, fontweight='bold')
                ax7.set_xlabel('Curriculum Level')
                ax7.set_ylabel('Combat Performance Score')
                ax7.grid(True, alpha=0.3)
            
            ax8 = fig.add_subplot(gs[3, :2])
            metrics_cols = ['success_rate', 'time_to_capture', 'deception_resistance_score', 
                           'combat_performance_score', 'episode_return_agent0']
            correlation_matrix = df[metrics_cols].corr()
            
            im = ax8.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            ax8.set_xticks(range(len(metrics_cols)))
            ax8.set_yticks(range(len(metrics_cols)))
            ax8.set_xticklabels([col.replace('_', ' ').title() for col in metrics_cols], rotation=45)
            ax8.set_yticklabels([col.replace('_', ' ').title() for col in metrics_cols])
            ax8.set_title('Metrics Correlation Matrix', fontsize=12, fontweight='bold')
            
            for i in range(len(metrics_cols)):
                for j in range(len(metrics_cols)):
                    text = ax8.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontweight='bold')
            
            plt.colorbar(im, ax=ax8, shrink=0.8)
            
            ax9 = fig.add_subplot(gs[3, 2:])
            win_stats = {
                'Agent 0 Wins': df['agent0_wins'].sum(),
                'Agent 1 Wins': df['agent1_wins'].sum(),
                'Draws': df['draws'].sum()
            }
            
            colors = ['skyblue', 'lightcoral', 'lightgray']
            wedges, texts, autotexts = ax9.pie(win_stats.values(), labels=win_stats.keys(), 
                                              autopct='%1.1f%%', colors=colors, startangle=90)
            ax9.set_title('Win/Loss/Draw Distribution', fontsize=12, fontweight='bold')
            
            for autotext in autotexts:
                autotext.set_fontweight('bold')
            
            plt.suptitle(f'{self.model} - Comprehensive Performance Metrics Analysis', 
                        fontsize=16, fontweight='bold')
            
            plt.savefig(f'{self.fig_dir}{self.model}_performance_metrics_analysis.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            print(f'Info: Performance metrics plot saved as {self.model}_performance_metrics_analysis.png')
            
        except Exception as e:
            print(f"Error plotting performance metrics: {e}")


    def plot_curriculum_performance_analysis(self, excel_filename=None):
        """Plot detailed curriculum performance analysis"""
        if excel_filename is None:
            import glob
            pattern = os.path.join(self.fig_dir, "combat_performance_metrics_*.xlsx")
            files = glob.glob(pattern)
            if not files:
                print("No performance metrics Excel file found")
                return
            excel_filename = max(files, key=os.path.getctime)
        
        try:
            df = pd.read_excel(excel_filename, sheet_name='Performance_Data')
            curriculum_df = pd.read_excel(excel_filename, sheet_name='Curriculum_Analysis')
            
            if 'curriculum_level' not in df.columns:
                print("No curriculum data found in Excel file")
                return
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'{self.model} - Curriculum Learning Performance Analysis', 
                        fontsize=16, fontweight='bold')
            
            ax1 = axes[0, 0]
            levels = sorted(df['curriculum_level'].unique())
            level_performance = []
            
            for level in levels:
                level_data = df[df['curriculum_level'] == level]['combat_performance_score']
                level_performance.append(level_data.mean())
            
            ax1.bar(range(len(levels)), level_performance, color='skyblue', alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Curriculum Level')
            ax1.set_ylabel('Average CPS')
            ax1.set_title('Performance by Curriculum Level')
            ax1.set_xticks(range(len(levels)))
            ax1.set_xticklabels([f'Level {int(l)}' for l in levels])
            ax1.grid(True, alpha=0.3, axis='y')
            
            ax2 = axes[0, 1]
            level_success = []
            for level in levels:
                level_data = df[df['curriculum_level'] == level]['success_rate']
                level_success.append(level_data.mean())
            
            ax2.bar(range(len(levels)), level_success, color='lightgreen', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Curriculum Level')
            ax2.set_ylabel('Average Success Rate')
            ax2.set_title('Success Rate by Curriculum Level')
            ax2.set_xticks(range(len(levels)))
            ax2.set_xticklabels([f'Level {int(l)}' for l in levels])
            ax2.grid(True, alpha=0.3, axis='y')
            
            ax3 = axes[0, 2]
            level_capture_time = []
            for level in levels:
                level_data = df[df['curriculum_level'] == level]['time_to_capture']
                level_capture_time.append(level_data.mean())
            
            ax3.bar(range(len(levels)), level_capture_time, color='orange', alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Curriculum Level')
            ax3.set_ylabel('Average Time to Capture')
            ax3.set_title('Capture Time by Curriculum Level')
            ax3.set_xticks(range(len(levels)))
            ax3.set_xticklabels([f'Level {int(l)}' for l in levels])
            ax3.grid(True, alpha=0.3, axis='y')
            
            ax4 = axes[1, 0]
            level_deception = []
            for level in levels:
                level_data = df[df['curriculum_level'] == level]['deception_resistance_score']
                level_deception.append(level_data.mean())
            
            ax4.bar(range(len(levels)), level_deception, color='lightcoral', alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Curriculum Level')
            ax4.set_ylabel('Average Deception Resistance')
            ax4.set_title('Deception Resistance by Level')
            ax4.set_xticks(range(len(levels)))
            ax4.set_xticklabels([f'Level {int(l)}' for l in levels])
            ax4.grid(True, alpha=0.3, axis='y')
            
            ax5 = axes[1, 1]
            level_counts = df['curriculum_level'].value_counts().sort_index()
            ax5.bar(range(len(level_counts)), level_counts.values, color='lightblue', alpha=0.7, edgecolor='black')
            ax5.set_xlabel('Curriculum Level')
            ax5.set_ylabel('Number of Episodes')
            ax5.set_title('Episodes per Curriculum Level')
            ax5.set_xticks(range(len(level_counts)))
            ax5.set_xticklabels([f'Level {int(l)}' for l in level_counts.index])
            ax5.grid(True, alpha=0.3, axis='y')
            
            ax6 = axes[1, 2]
            colors = plt.cm.tab10(np.linspace(0, 1, len(levels)))
            
            for i, level in enumerate(levels):
                level_data = df[df['curriculum_level'] == level].reset_index(drop=True)
                if len(level_data) > 1:
                    ax6.plot(range(len(level_data)), level_data['combat_performance_score'], 
                            color=colors[i], label=f'Level {int(level)}', alpha=0.7, linewidth=2)
            
            ax6.set_xlabel('Episode within Level')
            ax6.set_ylabel('Combat Performance Score')
            ax6.set_title('Performance Evolution within Levels')
            ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax6.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{self.fig_dir}{self.model}_curriculum_analysis.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            print(f'Info: Curriculum analysis plot saved as {self.model}_curriculum_analysis.png')
            
        except Exception as e:
            print(f"Error plotting curriculum analysis: {e}")
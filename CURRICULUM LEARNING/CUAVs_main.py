

from Simulation.Utils.utils import setup_parser
from Simulation.CUAVs_sim import Simulation

def main():
    train_mode = True
    render = False
    train_episodes = 1500

    args = setup_parser()

    sim = Simulation(args, train_mode, train_episodes, render)
    sim.run_simulation()

    try:
        from Simulation.Utils.utils import DataVisualization

        train_data_visual = DataVisualization(
            train_episodes, 
            [], 
            sim.agent.model_name, 
            sim.agent.train_data_filename
        )

        print("\nGenerating performance metrics visualizations...")
        train_data_visual.plot_performance_metrics()
        train_data_visual.plot_curriculum_performance_analysis()

        performance_summary = sim.agent.performance_metrics.get_current_performance_summary()
        print(f"\n{'='*80}")
        print(f"FINAL TRAINING SUMMARY")
        print(f"{'='*80}")
        print(f"Model: {sim.agent.model_name}")
        print(f"Total Episodes: {performance_summary.get('total_episodes', 0)}")
        print(f"Final Combat Performance Score: {performance_summary.get('recent_avg_cps', 0):.3f}")
        print(f"Final Success Rate: {performance_summary.get('recent_success_rate', 0):.3f}")
        print(f"Final Avg Time to Capture: {performance_summary.get('recent_avg_capture_time', 0):.1f} steps")
        print(f"Final Deception Resistance: {performance_summary.get('recent_deception_resistance', 0):.3f}")
        print(f"Performance Data Saved: {sim.agent.performance_metrics.excel_filename}")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"Error generating performance visualizations: {e}")

    sim.close_simulation()

if __name__ == '__main__':
    main()
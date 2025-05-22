from Simulation.Utils.utils import setup_parser
from Simulation.CUAVs_sim import Simulation

def main():
    train_mode = True
    render = False
    train_episodes = 100

    sim = Simulation(args, train_mode, train_episodes, render)
    sim.run_simulation()
    sim.close_simulation()


if __name__ == '__main__':
    args = setup_parser()
    main()

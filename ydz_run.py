import argparse

from src import config
from src.SNerf_slam import SNerf_slam

def main():
    # set run I/O folder
    parser = argparse.ArgumentParser(description='Argumants for SNerf_slam')
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str, help='input folder, higher priority')
    parser.add_argument('--output_folder', type=str, help='input folder, higher priority')
    args = parser.parse_args()

    cfg = config.load_config(args.config, 'configs/SNerf_slam')

    slam = SNerf_slam(cfg, args)

    slam.run()

if __name__ == '__main__':
    main()

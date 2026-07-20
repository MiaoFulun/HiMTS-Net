from pathlib import Path
import argparse

from himts_net.runner import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HiMTS-Net.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML configuration file.")
    args = parser.parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()


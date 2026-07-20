from pathlib import Path
import argparse

from himts_net.runner import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate HiMTS-Net on the held-out test split."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML configuration file.",
    )
    args = parser.parse_args()
    run_evaluation(args.config)


if __name__ == "__main__":
    main()

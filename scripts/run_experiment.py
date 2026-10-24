"""Entry point: one config + one seed -> one run folder. Currently a stub that only sets up the run."""
import argparse

from commitgate.config import load_config
from commitgate.io_utils import RunLogger, make_run_dir
from commitgate.seed import set_seed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exp-name", default="dummy")
    p.add_argument("--method", default="b0_static")
    p.add_argument("--stream", default="s1_abrupt")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = p.parse_args()

    cfg = load_config(args.config, [*args.overrides, f"seed={args.seed}"])
    set_seed(cfg.seed)
    run_dir = make_run_dir("results", args.exp_name, args.method, args.stream, cfg.seed, cfg)
    logger = RunLogger(run_dir)
    logger.log("run_start", method=args.method, stream=args.stream)
    # TODO(Phase 5): build stream, load model, run the stream loop.
    logger.write_summary({"status": "stub"})
    print(f"run folder: {run_dir}")


if __name__ == "__main__":
    main()

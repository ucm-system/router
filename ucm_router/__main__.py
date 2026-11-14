"""命令行入口：python -m ucm_router --config config.yaml"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from ucm_router.config import load_config
from ucm_router.main import create_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="ucm-router", description="UCM PD Router")
    parser.add_argument("--config", required=True, help="path to router YAML config")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    config = load_config(args.config)
    app = create_app(config)
    uvicorn.run(app, host=config.server.host, port=config.server.port, log_level=args.log_level)


if __name__ == "__main__":
    main()

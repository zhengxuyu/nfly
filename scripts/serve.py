"""Watch the fly play in a browser.

    python scripts/serve.py --suite atari --game pong --checkpoint runs/ppo-atari-pong.pt
    python scripts/serve.py --suite classic --game cartpole --policy random     # no data needed
"""

from __future__ import annotations

import argparse
import dataclasses

from nfly.cli import add_agent_args, add_connectome_args
from nfly.viz import SessionConfig, serve


def main() -> None:
    p = argparse.ArgumentParser()
    add_connectome_args(p, subset="visual")
    add_agent_args(p)
    p.add_argument("--suite", default="atari")
    p.add_argument("--game", default="pong")
    p.add_argument("--policy", default="fly", choices=["fly", "random"])
    p.add_argument("--checkpoint")
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-anatomy", action="store_true", help="skip the 3-D brain activity stream")
    p.add_argument("--max-points", type=int, default=30000, help="neurons drawn in the 3-D view")
    args = p.parse_args()

    fields = {f.name for f in dataclasses.fields(SessionConfig)}
    cfg = SessionConfig(**{k: v for k, v in vars(args).items() if k in fields}, data_dir=args.data, anatomy=not args.no_anatomy)
    server = serve(cfg, args.host, args.port)
    print(f"nfly viz: {server.url}  ({cfg.suite}/{cfg.game}, policy={cfg.policy})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

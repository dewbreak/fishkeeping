"""Command-line entry point, safe local binding by default."""

import argparse

import uvicorn

from fishkeeping.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(prog="fishkeeping")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Start the memory HTTP API")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    args = parser.parse_args()
    settings = Settings()
    uvicorn.run(
        "fishkeeping.api:create_app",
        factory=True,
        host=args.host or settings.host,
        port=args.port or settings.port,
    )


if __name__ == "__main__":
    main()

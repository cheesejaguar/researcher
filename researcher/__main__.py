"""`python -m researcher` entry point — delegates to the Typer app."""

from researcher.cli import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()

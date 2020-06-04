import typer
from .pipeline import train

app = typer.Typer()

app.command()(train)

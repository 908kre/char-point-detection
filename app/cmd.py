import typer
from .pipeline import eda_bboxes, train

app = typer.Typer()

app.command()(eda_bboxes)

app.command()(train)

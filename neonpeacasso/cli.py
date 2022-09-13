import typer
import uvicorn
import importlib
from pathlib import Path

try:
    p = importlib.util.find_spec("neonpeacasso").origin.replace("__init__.py", "ckpt_path.txt")
except:
    p = "ckpt_path.txt"
wp = ""
if not Path(p).exists():
    while "ckpt" not in wp:
        wp = input("Weights not found! Please paste the correct path to the model.ckpt file: ")
    with open(p, "w", encoding="utf-8") as f:
        f.write(wp)
        f.close()
else:
    wp = open(p, "r", encoding="utf-8").readline()
print(f"Will load from {wp}.")
print(f"Tip: if you want to change the weights' path, go to {p} and replace the value.")


app = typer.Typer()


@app.command()
def ui(
    host: str = "127.0.0.1", port: int = 8081, workers: int = 1, reload: bool = True
):
    """
    Launch the neonpeacasso UI.Pass in parameters host, port, workers, and reload to override the default values.
    """
    uvicorn.run(
        "neonpeacasso.web.backend.app:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
    )


@app.command()
def list():
    print("list")


def run():
    app()


if __name__ == "__main__":
    app()

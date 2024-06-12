from src.helpers.menu import Menu
from src.helpers.decorators import timer
from lightning.pytorch import seed_everything
from src.config import config


@timer
def main() -> None:
    seed_everything(config.seed, workers=True)
    Menu({
        # "1": ("Train model", train_model),
    }).start(timeout=60)


if __name__ == '__main__':
    main()

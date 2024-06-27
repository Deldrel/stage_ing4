from lightning.pytorch import seed_everything

from src.config import config
from src.helpers.decorators import timer
from src.helpers.menu import Menu
from src.model_training.lightning_manager import lightning_manager
from src.preprocessing.data_creation import create_csv_files
from src.preprocessing.sequencer import sequence
from src.preprocessing.visualizer import visualize, create_accessibility_map
from src.preprocessing.get_accessibility import create_accessibility_df


@timer
def main() -> None:
    seed_everything(config.seed, workers=True)
    Menu({
        "1": ("Query Accessibility Data", create_accessibility_df),
        "2": ("Craft new data", create_csv_files),
        "3": ("Create sequences", sequence),
        "4": ("Train model", lightning_manager.train_model),
        "5": ("Show plots", visualize),
        "6": ("Create Maps", create_accessibility_map),
    }).start(timeout=20)


if __name__ == '__main__':
    main()

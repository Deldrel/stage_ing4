import signal
from typing import Callable, Dict, Optional


class Menu:
    """
    A simple menu class.
    Option "0" is reserved for exiting the menu.
    """

    def __init__(self, options: Optional[Dict[str, tuple[str, Callable]]] = None) -> None:
        self.menu_options = {"0": ("Exit", self.stop)}
        if options:
            if "0" in options:
                raise ValueError("Option '0' is reserved for exiting the menu")
            self.menu_options.update(options)
        self.running = False

    def __str__(self) -> str:
        """
        String representation of the menu.
        :return: str
        """
        return "\n" + "\n".join([f"{key}. {description}" for key, (description, _) in self.menu_options.items()])

    def start(self, timeout: int = 30) -> None:
        """
        Start the menu.
        :param timeout: timeout in seconds
        :return: None
        """
        self.running = True
        while self.running:
            print(self)
            try:
                self._set_alarm(timeout)
                choice = input(f"Choose an option (timeout in {timeout}s): ")
                signal.alarm(0)
                self.handle_choice(choice)
            except TimeoutError:
                print("No input received, exiting menu...")
                self.stop()
            finally:
                signal.alarm(0)

    def handle_choice(self, choice: str) -> None:
        """
        Handle the choice of the user and call the chosen function.
        :param choice: str key of the option dictionary.
        :return: None
        """
        if choice in self.menu_options:
            self.menu_options[choice][1]()
        else:
            print("Invalid option")

    def _set_alarm(self, timeout: int) -> None:
        """
        Set an alarm for the timeout.
        :param timeout: int timeout in seconds
        :return: None
        """
        signal.signal(signal.SIGALRM, self._handle_timeout)
        signal.alarm(timeout)

    def _handle_timeout(self, signum, frame) -> None:
        """
        Handle the timeout.
        :param signum:
        :param frame:
        :return: None
        """
        raise TimeoutError

    def stop(self) -> None:
        """
        Stop the menu.
        :return: None
        """
        self.running = False

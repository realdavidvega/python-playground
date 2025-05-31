import os
from getpass import getpass


# Helper function for setting environment variables
def set_env(var: str) -> str:
    if not os.environ.get(var):
        os.environ[var] = getpass(f"Please provide your {var}")

    return os.environ[var]

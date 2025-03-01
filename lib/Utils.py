import os
import sys
from typing import Optional
from dotenv import load_dotenv


def GetRoot() -> None:
    """
    Set the root directory of the project.
    Usage in Python script:
    GetRoot()
    root = os.environ['PROJECT_ROOT']
    :return:None
    """
    # Allow to use the ".env" file
    load_dotenv()

    abs_path = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.dirname(abs_path)
    os.environ['PROJECT_ROOT'] = project_root
    return None

def GetModule():
    GetRoot()

    # get and set the environment variables
    python_path = os.environ.get('PYTHONPATH')
    if python_path:
        for path in python_path.split(':'):
            path = str(os.environ['PROJECT_ROOT']) + path
            if path not in sys.path:
                sys.path.append(path)

def GetHfToken() -> Optional[str]:
    # Load environment variables
    GetRoot()

    # Get ".token" file
    fil_token = os.environ.get("PROJECT_ROOT") + os.getenv("HF_TOKEN")

    # Get HuggingFace token
    if os.path.exists(fil_token):
        with open(fil_token, "r") as fil_token:
            hf_token = fil_token.read().strip()
            command = f"export HUGGINGFACE_TOKEN={hf_token}"
            os.system(command)
            return ">>> HuggingFace token applied."
    return None

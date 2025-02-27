import os
import sys
from dotenv import load_dotenv


def get_root() -> None:
    """
    Set the root directory of the project.
    Usage in Python script:
    get_root()
    root = os.environ['PROJECT_ROOT']
    :return:None
    """
    # Allow to use the ".env" file
    load_dotenv()

    abs_path = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.dirname(abs_path)
    os.environ['PROJECT_ROOT'] = project_root
    return None

def get_module():
    get_root()

    # get and set the environment variables
    python_path = os.environ.get('PYTHONPATH')
    if python_path:
        for path in python_path.split(':'):
            path = str(os.environ['PROJECT_ROOT']) + path
            if path not in sys.path:
                sys.path.append(path)


if __name__ == '__main__':
    print(get_root())
    print(os.getenv('INPUT_PATH'))

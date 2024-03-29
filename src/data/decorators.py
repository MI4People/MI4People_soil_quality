import inspect
import importlib
from pathlib import Path
from functools import wraps
import pandas as pd
import numpy as np

import inspect
import importlib
from pathlib import Path
from functools import wraps
import numpy as np
import pandas as pd


# The decorator-based approach to testing and data validation is currently not used
# as it was deemed to time intensive to maintain for its benefits.


def find_decorated_functions_in_this_dir(decorator_name=None):
    """This parses all .py-files in the dir THIS FILE is located and
       returns a list of all decorated functions by parsing a string of the
       functions first line.
       Beware, changing dir structure may break this.

    Args:
        decorator_name (str, optional): If set, only returns functions decorated with a specific decorator.
        Keep @ in string. Defaults to None.

    Returns:
        list: list of tuples (function_name:str, function:function, decorator_name:str)
    """
    decorated_functions = list()
    dir = Path(__file__).parent.resolve()
    files = [x for x in dir.glob("*.py")]
    for file in files:
        print(file)
        file = Path(file)
        module_name = inspect.getmodulename(file)
        spec = importlib.util.spec_from_file_location(module_name, file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        functions = inspect.getmembers(mod, inspect.isfunction)
        for func in functions:
            func_lines = inspect.getsourcelines(func[1])[0]
            if func_lines[0].startswith("@"):
                if decorator_name:
                    if func_lines[0].strip("\n").strip() == decorator_name:
                        decorated_functions.append(
                            (*func, func_lines[0].strip("\n").strip())
                        )
                else:
                    decorated_functions.append(
                        (*func, func_lines[0].strip("\n").strip())
                    )

    return decorated_functions


def img_preprocessing_decorator(f):
    # This decorator has some assertions for all in- and outputs at runtime
    @wraps(f)
    def decorator(*args, **kwargs):
        # assert isinstance(out, np.array)
        # assert len(out.shape) == 3
        out = f(*args, **kwargs)
        assert isinstance(out, np.array)
        assert len(out.shape) == 3
        return out

    return decorator


def csv_preprocessing_decorator(f):
    # This decorator has some assertions for all in- and outputs at runtime
    @wraps(f)
    def decorator(*args, **kwargs):
        out = f(*args, **kwargs)
        assert isinstance(out, (pd.DataFrame, pd.Series))
        return out

    return decorator


def csv_ingestion_decorator(f):
    # This decorator has some assertions for all in- and outputs at runtime
    @wraps(f)
    def decorator(*args, **kwargs):
        out = f(*args, **kwargs)
        assert isinstance(out, (pd.DataFrame, pd.Series))
        return out

    return decorator


def img_ingestion_decorator(f):
    # This decorator has some assertions for all in- and outputs at runtime
    @wraps(f)
    def decorator(*args, **kwargs):
        assert True
        out = f(*args, **kwargs)
        assert isinstance(out, np.array)
        assert len(out.shape) == 3
        return out

    return decorator


def test_decorated_functions():
    functions = find_decorated_functions_in_this_dir()


if __name__ == "__main__":
    test_decorated_functions()

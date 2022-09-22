import inspect
import importlib
from pathlib import Path
from functools import wraps


def tdecorator(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        print(inspect.getsourcelines(f)[0][0])
        f.__name__ = f"test_{f.__name__}"
        return f(*args, **kwargs)

    return decorator


@tdecorator
def add(a, b):
    return a + b


def find_decorated_functions_in_parent_dir():
    decorated_functions = list()
    dir = Path(__file__).parent.resolve()
    files = [x for x in dir.glob("*.py")]
    for file in files:
        file = Path(file)
        module_name = inspect.getmodulename(file)
        spec = importlib.util.spec_from_file_location(module_name, file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        functions = inspect.getmembers(mod, inspect.isfunction)
        for func in functions:
            func_lines = inspect.getsourcelines(func[1])[0]
            if func_lines[0].startswith("@"):
                decorated_functions.append(func)
    print(decorated_functions)


if __name__ == "__main__":
    # print(add(3,6))
    find_decorated_functions_in_parent_dir()
# print(decorated_functions) # -> Muss wieder alle im Test-file aufrufen
# Möglich: parse alle im modul, adde alle mit decorator?

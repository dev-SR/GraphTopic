from rich.console import Console

console = Console()

# see the available choices for spinner:
# python -m rich.spinner
# https://rich.readthedocs.io/en/latest/console.html#status


def withLoader(cb, message="", spinner="point"):
    done = False
    returns = None
    with console.status(f"[bold yellow] {message}...", spinner=spinner) as s:
        while not done:
            returns = cb()
            done = True
    return returns


# def withLoaderWithParam(cb, param, message="", spinner='point'):
#     done = False
#     returns = None
#     with console.status(f"[bold yellow] {message}...", spinner=spinner) as s:
#         while not done:
#             returns = cb(*param)
#             done = True
#     return returns


def withLoaderWithParam(cb, param, message="", spinner="dots"):
    done = False
    returns = None
    with console.status(f"[bold yellow] {message}...", spinner=spinner) as s:
        while not done:
            returns = cb(**param)
            done = True
    return returns

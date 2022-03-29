"""Module to catch exceptions."""
from typing import Tuple


def print_detailed_exception(func_name: str,
                             sys_info: Tuple[any, ...],
                             expt: Exception) -> None:
    """prints detailed message of exception encountered.

    Args:
        func_name: name of the function the exception occurred.
        sys_info: object which contains information regarding the script, use it to obtain the line number.
        expt: Exception object passed.

    Returns: None.
    """
    print(f'Error in function: {func_name}\n' +
          f'line no: {int(sys_info[-1].tb_lineno)}\n' +
          f'args error: {expt.args}\n' +
          f'message error: {expt.__doc__}\n')

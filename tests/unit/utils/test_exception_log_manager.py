import pytest
import sys
from inspect import currentframe
from k_means.utils.exception_log_manager import print_detailed_exception


@pytest.mark.utils_exception_log_manager
def test_print_detailed_exception(capfd):
    try:
        print(x)
    except Exception as expt:
        print_detailed_exception(currentframe().f_code.co_name, sys.exc_info(), expt)
        out, err = capfd.readouterr()
        assert \
            out == 'Error in function: test_print_detailed_exception\n' + \
            'line no: 10\n' + \
            'args error: ("name \'x\' is not defined",)\n' + \
            'message error: Name not found globally.\n\n'

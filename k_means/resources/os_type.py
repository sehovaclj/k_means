"""Module that identifies which os we are working on."""
import os
import sys

global dir_slash

if os.name == 'posix':
    dir_slash = '/'
elif os.name == 'nt':
    dir_slash = '\\'
else:
    print('We are neither running on a linux or windows OS, exiting')
    sys.exit()

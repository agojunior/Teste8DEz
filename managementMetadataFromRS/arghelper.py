"""
Set operations for validate input and output files or directories.
:Contains:
 is_valid_file,
 is_valid_directory

:Notes:
This module enable verify if input and output files and directories passed to ArgParse arguments are valid. The
code was adapted from https://codereview.stackexchange.com posted for Matthew Rankin.
:Author: Matthew Rankin
"""
# Fonte: https://codereview.stackexchange.com/questions/28608/checking-if-cli-arguments-are-valid-files-directories-in-python

import os


def is_valid_file(parser, arg):
    if not os.path.isfile(arg):
        parser.error('The file {} does not exist!'.format(arg))
    else:
        # File exists so return the filename
        return arg


def is_valid_directory(parser, arg):
    if not os.path.isdir(arg):
        parser.error('The directory {} does not exist!'.format(arg))
    else:
        # File exists so return the directory
        return arg

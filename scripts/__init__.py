# This file makes scripts/ an importable Python package.
#
# It must exist for TerraTorch's custom_modules_path mechanism to work.
# TerraTorch's cli_tools.py does:
#
#     workdir = custom_modules_path.parents[0]     # your project root
#     module_dir = custom_modules_path.name         # "scripts"
#     sys.path.insert(0, str(workdir))
#     importlib.import_module(module_dir)           # import_module("scripts")
#
# importlib.import_module("scripts") only succeeds if scripts/ contains
# an __init__.py — otherwise Python doesn't recognise it as a package,
# and you get an ImportError even though the directory and files exist.
#
# RENAME THIS FILE when you copy it into your project:
#   scripts__init__.py  -->  scripts/__init__.py
#
# (It's named with a double-underscore here only because this sandbox
# can't create a literal scripts/ subdirectory for you to download —
# once it's in your own project, it must sit at scripts/__init__.py.)
#
# It can stay completely empty. Nothing needs to go in it.

"""
Root conftest.py — ensures the project root is on sys.path so that
`import src.*` and `import config` both resolve correctly when pytest
is invoked from any working directory.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

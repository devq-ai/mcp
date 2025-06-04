#!/usr/bin/env python3

import sys
import importlib

print(f"Python version: {sys.version}")
print(f"Python path: {sys.executable}")

try:
    import surrealdb
    print(f"SurrealDB version: {surrealdb.__version__ if hasattr(surrealdb, '__version__') else 'unknown'}")
    print(f"SurrealDB module: {surrealdb.__file__}")
    print(f"SurrealDB dir: {dir(surrealdb)}")
except ImportError as e:
    print(f"Error importing SurrealDB: {e}")

try:
    from surrealdb import Surreal
    print(f"Surreal class dir: {dir(Surreal)}")
except ImportError as e:
    print(f"Error importing Surreal: {e}")
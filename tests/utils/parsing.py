"""
tests/utils/parsing.py
----------------------
Includes the building of parser specifically the argument parser that
abstract the parser construction. In additional the @mainrunner is a
decoration to also abstract the logic away from the actual testing 
scripts for simplicity usage and modularity.
"""
import argparse
import sys
import traceback

from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List


@dataclass
class CommandSpec:
    name    : str
    help    : str
    handler : Callable
    args    : List[Dict[str,Any]]=field(default_factory=list)

    def __post_init__(self):
        
        if not self.name.isidentifier():
            raise ValueError(f"Command name `{self.name}` must be valid")
        
        if not callable(self.handler):
            raise TypeError(f"{self.handler} must be callable")


def build_parser(commands: List[CommandSpec]) -> argparse.Namespace:
    """
    Takes different functions into a wrapped Command Specifics with a Callable
    where name and function arguments are passed and compressed into a subparser
    which is called in Cli scripts
    """
    parser = argparse.ArgumentParser(description="Logger, debug Cli")
    sub = parser.add_subparsers(dest="command", required=True)

    for command in commands:
        p = sub.add_parser(command.name, help=command.help)
        
        for argument in command.args:
            p.add_argument(*argument["flags"], **argument["kwargs"])
        
        p.set_defaults(_handler=command.handler)
    
    return parser


def mainrunner(f: Callable) -> Callable:
    @wraps(f)
    def wrapper(*args, **kwargs):
        try: 
            return f(*args, **kwargs)
        
        except KeyboardInterrupt:
            print("\nAborted by user")
            raise
        
        except Exception as e:
            print("\nTest failed:\t{e}")
            traceback.print_exc()
            raise
    
    return wrapper
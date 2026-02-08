"""
SynthERA-835: Synthetic X12 835 EDI Generator

A research tool for generating synthetic Electronic Remittance Advice (ERA)
files in X12 835 format for healthcare denial classification benchmarking.

Author: Velden Health Research Team
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Velden Health"

from .generator import ERA835Generator
from .carc_parser import CARCParser
from .rarc_parser import RARCParser

__all__ = ["ERA835Generator", "CARCParser", "RARCParser"]

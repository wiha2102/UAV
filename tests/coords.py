"""
tests/coords.py
---------------
Testing the conversion of the coordinates systems as well as testing the 
performance of subtracting and adding vectors, also measures the approximation
accuracy.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import time
import numpy as np

from src.math import coords
from tests.utils.parsing import CommandSpec, build_parser, mainrunner
from tests.utils.timing import Timer

from typing import Tuple


# ============================================================
#   Generate vectors
# ============================================================

def gen_cart(n: int, d:int=3, s:int=42) -> np.ndarray:
    np.random.seed(s)
    if d == 2:
        v = np.random.randn(n,3).astype(np.float64)
        v[:,2]=0.0
    elif d == 3: v = np.random.randn(n,3).astype(np.float64)
    else: raise ValueError(f"{d} unsupported dim")
    return v

def gen_sph(n:int, s:int=42) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    np.random.seed(s)
    return (
        np.random.uniform(0.1,100.0,n).astype(np.float64),
        np.random.uniform(-180.0,180.0,n).astype(np.float64),
        np.random.uniform(0.0,180.0,n).astype(np.float64)
    )


# ============================================================
#   Test Methods
# ============================================================

def test_cart_to_sph(args: argparse.Namespace):
    v = gen_cart(args.samples, args.dim, args.seed)
    with Timer("Cartesian to Spherical:\t", print_result=True):
        _,_,_ = coords.cartesian_to_spherical(v)

def test_sph_to_cart(args: argparse.Namespace):
    r,p,t = gen_sph(args.samples, args.seed)
    with Timer("Spherical to Cartesian:\t", print_result=True):
        _ = coords.spherical_to_cartesian(r, p, t)


def test_cart_to_sph_r(args: argparse.Namespace):
    u = gen_cart(args.samples, args.dim, args.seed)
    with Timer("Cartesian to Spherical:\t", print_result=True):
        r,p,t = coords.cartesian_to_spherical(u)
    
    with Timer("Spherical to Cartesian:\t", print_result=True):
        v = coords.spherical_to_cartesian(r, p, t)
    
    print("---------------======== Differences ========---------------")
    print(v - u)

def test_sph_to_cart_r(args: argparse.Namespace):
    a,b,c = gen_sph(args.samples, args.seed)
    with Timer("Spherical to Cartesian:\t", print_result=True):
        v = coords.spherical_to_cartesian(a,b,c)
    with Timer("Cartesian to Spherical:\t", print_result=True):
        d,e,f = coords.cartesian_to_spherical(v)
    
    print("---------------======== Differences ========---------------")
    print(f"Radius:\n{d-a}")
    print(f"Phi:\n{e-b}")
    print(f"Theta:\n{f-c}")

def test_add_angles(args: argparse.Namespace):
    _, p1, t1 = gen_sph(args.samples, args.seed)
    _, p2, t2 = gen_sph(args.samples, args.seed)

    with Timer("Add Angles:\t", print_result=True):
        _, _ = coords.add_angles(phi0=p1,phi1=p2,theta0=t1,theta1=t2)

def test_sub_angles(args: argparse.Namespace):
    _, p1, t1 = gen_sph(args.samples, args.seed)
    _, p2, t2 = gen_sph(args.samples, args.seed)

    with Timer("Add Angles:\t", print_result=True):
        _, _ = coords.sub_angles(phi0=p1,phi1=p2,theta0=t1,theta1=t2)

# ============================================================
#   Mainrunner
# ============================================================

D = [
    {"flags":["--samples","-n"],"kwargs":{"type":int,"default":1000}},
    {"flags":["--seed","-s"],"kwargs":{"type":int,"default":42}},
    {"flags":["--dim","-d"],"kwargs":{"type":int,"default":3}}
]

@mainrunner
def main():
    p = build_parser([
        CommandSpec("cart2sph","conversion cart to sph",test_cart_to_sph,[*D]),
        CommandSpec("sph2cart","conversion sph to cart",test_sph_to_cart,[*D]),
        # CommandSpec("cart2sph-r","conversion cart to sph round",test_cart_to_sph_r,[*D]),
        # CommandSpec("sph2cart-r","conversion cart to sph round",test_sph_to_cart_r,[*D]),
        CommandSpec("add", "add angles", test_add_angles, [*D]),
        CommandSpec("sub", "sub angles", test_sub_angles, [*D])
    ])
    args = p.parse_args()
    args._handler(args)


if __name__ == "__main__":
    main()

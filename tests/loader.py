import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))


import argparse
from data.loader import DataLoader, shuffle_and_split
from tests.utils.parsing import CommandSpec, build_parser, mainrunner
from tests.utils.timing import Timer



# ============================================================
#       Testing Methods
# ============================================================

def test_load_data(args: argparse.Namespace):
    """ Testing loading dataset """
    loader = DataLoader()
    with Timer("Loading data:\t", print_result=True):
        data = loader.load(args.dataset)


def test_data_shape(args: argparse.Namespace):
    loader = DataLoader()
    data = loader.load([args.dataset])

    fkey = list(data.keys())[0] if data else None
    rows, cols = len(data[fkey]) if fkey else 0, len(data)

    print(f"Data shape: {rows} rows × {cols} columns")
    print(f"Required columns ({len(loader.REQUIRED_COLUMNS)}): {loader.REQUIRED_COLUMNS}")
    print(f"Loaded columns ({cols}): {list(data.keys())}")

    print("\nData types:")
    for key, array in data.items():
        print(f"  {key}: {array.dtype}, shape: {array.shape}")


# ============================================================
#       Mainrunner
# ============================================================

DATA = [{"flags": ["--dataset"],"kwargs": {"type":str,"default":"uav_london/train.csv"}}]

@mainrunner
def main():
    p = build_parser([
        CommandSpec("load","Testing load data",test_load_data,[*DATA]),
        CommandSpec("shape","Testing data shape",test_data_shape,[*DATA])
    ])
    args = p.parse_args()
    args._handler(args)



if __name__ == "__main__":
    main()

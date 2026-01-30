#!/usr/bin/env python3
# This tool parses the test case config (/tests/config.yaml) for selecting
# test cases.

import argparse
import os
import sys

import yaml

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--group", help="Name of test groups to run",
                        type=str, required=True)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    cwd = os.path.dirname(os.path.realpath(__file__))
    root = os.path.dirname(cwd)
    cfg_path = os.path.join(root, "tests", "config.yaml")
    cfg = {}
    with open(cfg_path, "r") as file:
        try:
            cfg = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(f"Error occurred when reading config: {exc}",
                  file=sys.stderr)
            return -1

    test_root = cfg.get("unit-tests", {}).get("root", None)
    if test_root is None:
        print("tests/config.yaml malformed - missing 'root' under 'unit-tests'",
              file=sys.stderr)
        return -1

    files = cfg.get("unit-tests", {}).get("files", [])
    if len(files) == 0:
        print("tests/config.yaml malformed - missing 'files' under 'unit-tests'",
              file=sys.stderr)
        return -1

    for f in files:
        if not isinstance(f, dict):
            # ignore invalid entry
            continue
        path = f.get("path", "")
        labels = f.get("labels", [])
        if not isinstance(path, str) or not isinstance(labels, list):
            print(f"tests/config.yaml malformed - {f}", file=sys.stderr)
            continue
        if args.group in labels:
            if args.group in f.get("extra_args", {}):
                # Some files are tested with different configuration variants
                variants = f["extra_args"].get(args.group, [])
                for v in variants:
                    print(f"{path} {v}")
            else:
                    print(f"{path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

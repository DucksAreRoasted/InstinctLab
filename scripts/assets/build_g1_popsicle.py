#!/usr/bin/env python3

"""Build the immutable G1 popsicle USD with analytic capsule collisions."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--variant",
    choices=("all", "base", "shoe"),
    default="all",
    help="Asset variant to build (default: all).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from instinctlab.assets.g1_popsicle_asset import G1_POPSICLE_SHOE_SPEC, G1_POPSICLE_SPEC, asset_digest, build_asset


def main():
    specs = {
        "all": (G1_POPSICLE_SPEC, G1_POPSICLE_SHOE_SPEC),
        "base": (G1_POPSICLE_SPEC,),
        "shoe": (G1_POPSICLE_SHOE_SPEC,),
    }
    for spec in specs[args_cli.variant]:
        asset_path = build_asset(spec)
        print(f"Asset variant: {spec.cache_namespace}")
        print(f"Asset digest: {asset_digest(spec)}")
        print(f"Final USD: {asset_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2025-09-17
Purpose: Create preview images
"""

import argparse
import contextlib
import io
import os
import constants
import subprocess
import tempfile
import time
import MDAnalysis as mda
import sys
import warnings
from PIL import Image
from typing import NamedTuple, Iterator
from playwright.sync_api import sync_playwright

warnings.filterwarnings("ignore")

class Args(NamedTuple):
    """ Command-line arguments """
    structure: str
    trajectory: str
    out_file: str
    height: int
    width: int


# --------------------------------------------------
def get_args() -> Args:
    """ Get command-line arguments """

    parser = argparse.ArgumentParser(
        description='Create preview images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-s',
                        '--structure',
                        help='Structure file',
                        metavar='FILE',
                        required=True)

    parser.add_argument('-t',
                        '--trajectory',
                        help='Trajectory file',
                        metavar='FILE',
                        required=True)

    parser.add_argument('-o',
                        '--out-file',
                        help='Output file',
                        metavar='FILE',
                        required=True)

    parser.add_argument('-H',
                        '--height',
                        help='Image height',
                        metavar='INT',
                        default=500)

    parser.add_argument('-w',
                        '--width',
                        help='Image width',
                        metavar='INT',
                        default=500)

    args = parser.parse_args()

    return Args(structure=args.structure,
                trajectory=args.trajectory,
                out_file=args.out_file,
                height=args.height,
                width=args.width)


# --------------------------------------------------
@contextlib.contextmanager
def ensure_display() -> Iterator[None]:
    """Start Xvfb if no DISPLAY is set, and clean up afterward."""
    for num in range(99, 200):
        lock = f"/tmp/.X{num}-lock"
        if not os.path.exists(lock):
            display = f":{num}"
            proc = subprocess.Popen(
                ["Xvfb", display, "-screen", "0", "1024x768x24"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for _ in range(50):
                if os.path.exists(lock):
                    break
                time.sleep(0.1)
            else:
                proc.terminate()
                continue
            prev_display = os.environ.get("DISPLAY")
            os.environ["DISPLAY"] = display
            try:
                yield
            finally:
                proc.terminate()
                proc.wait()
                if prev_display is None:
                    del os.environ["DISPLAY"]
                else:
                    os.environ["DISPLAY"] = prev_display
            return

    raise RuntimeError("No free display number found for Xvfb")


# --------------------------------------------------
def main() -> None:
    """ Make a jazz noise here """

    args = get_args()
    script_dir = os.path.dirname(sys.argv[0])
    universe = mda.Universe(args.structure, args.trajectory)
    num_frames = len(universe.trajectory)
    width = 500
    height = 500

    protein = universe.select_atoms("protein")
    is_cg = protein.n_residues > 0 and protein.n_atoms == protein.n_residues

    with ensure_display(), sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=[
                "--enable-unsafe-swiftshader",
                "--ignore-gpu-blocklist",
                "--no-sandbox",
                "--disable-setuid-sandbox",
            ],
        )
        page = browser.new_page()
        page.on("console", lambda msg: print(f"[browser {msg.type}] {msg.text}", file=sys.stderr))
        page.on("pageerror", lambda err: print(f"[browser error] {err}", file=sys.stderr))
        page.set_viewport_size({"width": width, "height": height})
        page.set_content(constants.HTML)
        page.add_style_tag(content=constants.CSS)
        page.add_script_tag(path=os.path.join(script_dir, "ngl.js"))
        page.set_input_files("#structure-element", args.structure)
        page.set_input_files("#trajectory-element", args.trajectory)
        page.add_script_tag(content=constants.JS)
        page.set_default_timeout(600 * 10**3)
        page.evaluate(f"window.isCoarseGrained = {'true' if is_cg else 'false'}")
        page.evaluate("loadStage()")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        page.screenshot(path=tmp.name, omit_background=True)
        screenshot_img = make_transparent(Image.open(tmp.name))
        screenshot_img.save(args.out_file)
        browser.close()

    print(f"Wrote '{args.out_file}'")


# --------------------------------------------------
def make_transparent(image: Image.Image):
    """Make image transparent"""

    image = image.convert("RGBA")
    data = image.load()
    width, height = image.size

    for y in range(height):
        for x in range(width):
            item = data[x, y]  # type: ignore
            if all(i == 255 for i in item):
                data[x, y] = (255, 255, 255, 0)  # type: ignore

    return image


# --------------------------------------------------
if __name__ == '__main__':
    main()

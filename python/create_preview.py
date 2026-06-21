#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2025-09-17
Purpose: Create preview images

Renders the thumbnail with Mol* (molstar) the same way the elm-mdrepo web
viewer does -- Viewer.create + loadTrajectory(minimal.pdb + sampled.xtc) -- so
the depiction and chain coloring match the site. White background. The molstar
build is checked in alongside this script as molstar.js / molstar.css.
"""

import argparse
import base64
import contextlib
import os
import subprocess
import sys
import time
import warnings
from typing import NamedTuple, Iterator
from playwright.sync_api import sync_playwright

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
MOLSTAR_JS = os.path.join(HERE, "molstar.js")
MOLSTAR_CSS = os.path.join(HERE, "molstar.css")


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
                        type=int,
                        default=500)

    parser.add_argument('-w',
                        '--width',
                        help='Image width',
                        metavar='INT',
                        type=int,
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


# Mirror the site: Viewer.create + loadTrajectory(model + coordinates), with all
# panels/overlays off so the screenshot is just the molecule on a white
# background.
LOAD_JS = r"""
async ({pdb, xtcB64}) => {
  const viewer = await molstar.Viewer.create('app', {
    layoutIsExpanded: false,
    layoutShowControls: false,
    layoutShowLeftPanel: false,
    layoutShowSequence: false,
    layoutShowLog: false,
    layoutShowRemoteState: false,
    viewportShowExpand: false,
    viewportShowAnimation: false,
    viewportShowSelectionMode: false,
    viewportShowControls: false,
    pdbProvider: 'rcsb',
    emdbProvider: 'rcsb',
  });
  window.molViewer = viewer;
  viewer.plugin.canvas3d?.setProps({
    renderer: { backgroundColor: 0xffffff },
    camera: { helper: { axes: { name: 'off', params: {} } } },
  });
  if (xtcB64) {
    const bin = atob(xtcB64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    await viewer.loadTrajectory({
      model: { kind: 'model-data', data: pdb, format: 'pdb' },
      coordinates: { kind: 'coordinates-data', data: bytes, format: 'xtc' },
    });
  } else {
    await viewer.loadStructureFromData(pdb, 'pdb');
  }
  viewer.plugin.managers.camera.reset();
  return true;
}
"""


# --------------------------------------------------
def main() -> None:
    """ Make a jazz noise here """

    args = get_args()
    width = int(args.width)
    height = int(args.height)

    with open(args.structure) as fh:
        pdb = fh.read()

    xtc_b64 = None
    if args.trajectory and os.path.isfile(args.trajectory):
        with open(args.trajectory, "rb") as fh:
            xtc_b64 = base64.b64encode(fh.read()).decode("ascii")

    with ensure_display(), sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=[
                "--enable-unsafe-swiftshader",
                "--ignore-gpu-blocklist",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        page = browser.new_page(viewport={"width": width, "height": height})
        page.on("pageerror", lambda err: print(f"[browser error] {err}", file=sys.stderr))
        page.set_content(
            "<html><head></head><body style='margin:0'>"
            f"<div id='app' style='position:absolute;inset:0;width:{width}px;height:{height}px'></div>"
            "</body></html>"
        )
        page.add_style_tag(path=MOLSTAR_CSS)
        page.add_style_tag(content=".msp-viewport-controls,.msp-viewport-top-left-controls{display:none!important}")
        page.add_script_tag(path=MOLSTAR_JS)
        page.set_default_timeout(180 * 10**3)
        page.evaluate(LOAD_JS, {"pdb": pdb, "xtcB64": xtc_b64})
        page.wait_for_timeout(2500)
        # Screenshot the molstar canvas directly (no HTML overlays).
        canvas = page.query_selector("canvas.msp-canvas") or page.query_selector("canvas")
        canvas.screenshot(path=args.out_file)
        browser.close()

    print(f"Wrote '{args.out_file}'")


# --------------------------------------------------
if __name__ == '__main__':
    main()

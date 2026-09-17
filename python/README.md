# Setup

Install [uv](https://docs.astral.sh/uv/#installation)

Then:

```
uv sync
```

`uv sync` installs exactly what `uv.lock` pins. Do not use
`uv add --requirements ...`: that re-resolves every dependency and moves the
lock, which is how this environment and `utils/python` drift apart on shared
libraries such as MDAnalysis. Both are meant to hold the same versions.

To install `playwright`:

```
source .venv/bin/activate && playwright install
```

Programs can be run with:

* `uv run <program>` 
* or first do `source .venv/bin/activate` and execute directly

# Author

Ken Youens-Clark <kyclark@arizona.edu>

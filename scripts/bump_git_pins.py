"""Move the Git pins of Volumential's dependencies, then relock.

The inducer packages Volumential needs have no usable release, so
``pyproject.toml`` declares each of them as a direct reference pinned to a
commit, ``name @ git+https://github.com/inducer/<repo>.git@<commit>``. That
pin is what an install of Volumential gets, ``uv.lock`` records the same
commit, and CI tests it. This script is how the pins move.

With no package named, every pin moves to the head of its repository's
default branch, as ``git ls-remote <url> HEAD`` reports it. Name packages to
move only those, and write ``name=<commit>`` to move one to a given commit
instead of the head. The script rewrites the pins in ``pyproject.toml`` and
then runs ``uv lock``, which resolves the new commits and whatever they
require; ``git diff`` shows both files.

Usage::

    python scripts/bump_git_pins.py                  # every pin to its head
    python scripts/bump_git_pins.py sumpy pytential  # these two to their heads
    python scripts/bump_git_pins.py loopy=<commit>   # loopy to that commit
    python scripts/bump_git_pins.py --dry-run        # print the moves only
    python scripts/bump_git_pins.py --no-lock        # pyproject.toml only

``--dry-run`` writes nothing. ``--no-lock`` rewrites ``pyproject.toml`` and
leaves ``uv.lock`` alone, which is what the weekly dependency-heads job of
``CI Full`` uses: it installs the result with ``uv pip install``, which does
not read the lock.
"""

import argparse
import re
import subprocess
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path


# A pinned Git requirement: ``name @ git+<url>@<rev>``, optionally followed by
# an environment marker.  Only a full commit hash counts as a pin; a branch or
# tag name would move under the package without a change here.
_GIT_REQUIREMENT = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)\s*@\s*"
    r"git\+(?P<url>[^@\s;]+?)(?:@(?P<rev>[^\s;@]+))?"
    r"\s*(?:;.*)?$"
)
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


@dataclass
class Pin:
    """One Git dependency and the commit ``pyproject.toml`` pins it to."""

    name: str
    url: str
    rev: str
    #: Where the requirement occurs: ``dependencies`` or ``extra <name>``.
    places: list[str] = field(default_factory=list)


def normalize(name: str) -> str:
    """Return the PEP 503 normal form of a distribution name."""

    return re.sub(r"[-_.]+", "-", name).lower()


def read_pins(pyproject: dict) -> dict[str, Pin]:
    """Collect the Git pins of ``[project]``, keyed by normalized name.

    Raises :exc:`SystemExit` for a Git requirement that is not pinned to a
    full commit, or for a package pinned to two different commits.
    """

    project = pyproject["project"]
    groups = [("dependencies", project.get("dependencies", []))]
    groups += [
        (f"extra {extra}", requirements)
        for extra, requirements in project.get(
            "optional-dependencies", {}).items()
    ]

    pins: dict[str, Pin] = {}
    for place, requirements in groups:
        for requirement in requirements:
            if "git+" not in requirement:
                continue
            match = _GIT_REQUIREMENT.match(requirement)
            if match is None:
                raise SystemExit(f"cannot parse the Git requirement "
                                 f"{requirement!r} ({place})")
            rev = match["rev"] or ""
            if not _COMMIT.match(rev):
                raise SystemExit(f"{requirement!r} ({place}) is not pinned to "
                                 "a full commit hash")

            key = normalize(match["name"])
            pin = pins.setdefault(
                key, Pin(match["name"], match["url"], rev))
            if (pin.url, pin.rev) != (match["url"], rev):
                raise SystemExit(f"{match['name']} is pinned twice, to "
                                 f"{pin.url}@{pin.rev} and "
                                 f"{match['url']}@{rev}")
            pin.places.append(place)

    return pins


def remote_head(url: str) -> str:
    """Return the commit at the head of the default branch of ``url``."""

    result = subprocess.run(
        ["git", "ls-remote", "--exit-code", url, "HEAD"],
        check=False, capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(f"git ls-remote {url} HEAD failed "
                         f"(exit {result.returncode}): {result.stderr.strip()}")
    rev = result.stdout.split()[0]
    if not _COMMIT.match(rev):
        raise SystemExit(f"git ls-remote {url} HEAD printed {result.stdout!r}")
    return rev


def parse_targets(args: list[str], pins: dict[str, Pin]) -> dict[str, str | None]:
    """Map each package to move to its new commit, ``None`` for the head."""

    if not args:
        return dict.fromkeys(pins)

    targets: dict[str, str | None] = {}
    for arg in args:
        name, _, rev = arg.partition("=")
        key = normalize(name)
        if key not in pins:
            raise SystemExit(f"{name} has no Git pin in pyproject.toml; the "
                             f"pinned packages are {', '.join(sorted(pins))}")
        if rev and not _COMMIT.match(rev):
            raise SystemExit(f"{arg}: give the full 40-character commit hash")
        targets[key] = rev or None
    return targets


def compare_url(pin: Pin, new_rev: str) -> str:
    """Return a link to the commits between the old and the new pin."""

    base = pin.url.removesuffix(".git")
    if base.startswith("https://github.com/"):
        return f"{base}/compare/{pin.rev[:12]}...{new_rev[:12]}"
    return base


def main() -> None:
    """Move the pins named on the command line and relock."""

    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "packages", nargs="*", metavar="NAME[=COMMIT]",
        help="packages to move (default: all), optionally to a given commit")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the moves and write nothing")
    parser.add_argument(
        "--no-lock", action="store_true",
        help="rewrite pyproject.toml but do not run uv lock")
    parser.add_argument(
        "--pyproject", type=Path,
        default=Path(__file__).resolve().parent.parent / "pyproject.toml",
        help="the pyproject.toml to rewrite (default: this repository's)")
    args = parser.parse_args()

    text = args.pyproject.read_text()
    pins = read_pins(tomllib.loads(text))
    targets = parse_targets(args.packages, pins)

    heads = [key for key, rev in targets.items() if rev is None]
    with ThreadPoolExecutor(max_workers=8) as pool:
        resolved = dict(zip(
            heads, pool.map(lambda key: remote_head(pins[key].url), heads),
            strict=True))
    new_revs = {key: rev or resolved[key] for key, rev in targets.items()}

    width = max(len(pins[key].name) for key in new_revs)
    for key, new_rev in sorted(new_revs.items()):
        pin = pins[key]
        if new_rev == pin.rev:
            print(f"{pin.name:<{width}}  {pin.rev[:12]}  unchanged")
            continue
        print(f"{pin.name:<{width}}  {pin.rev[:12]} -> {new_rev[:12]}  "
              f"{compare_url(pin, new_rev)}")
        old = f"git+{pin.url}@{pin.rev}"
        if text.count(old) != len(pin.places):
            raise SystemExit(f"{args.pyproject}: expected {len(pin.places)} "
                             f"occurrence(s) of {old}, found {text.count(old)}")
        text = text.replace(old, f"git+{pin.url}@{new_rev}")

    if args.dry_run:
        print("dry run: nothing written")
        return

    args.pyproject.write_text(text)
    if args.no_lock:
        return

    # The table above goes out before uv's own output, not after it.
    sys.stdout.flush()
    status = subprocess.run(["uv", "lock"], cwd=args.pyproject.parent,
                            check=False).returncode
    if status:
        print(f"uv lock failed; {args.pyproject} has the new pins and uv.lock "
              "the old ones (git checkout pyproject.toml undoes the move)",
              file=sys.stderr)
        sys.exit(status)


if __name__ == "__main__":
    main()

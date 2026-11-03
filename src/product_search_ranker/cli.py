"""`psr` command line. One subcommand per pipeline step; each prints what it wrote."""

import argparse

from product_search_ranker import config


def _fetch(args):
    from product_search_ranker.data.fetch import fetch

    fetch(if_changed=args.if_changed)


def _prepare(args):
    from product_search_ranker.data.prepare import prepare

    prepare(fixtures=config.fixtures_mode(args.fixtures))


def _make_fixture(args):
    from product_search_ranker.data.fixture import make_fixture

    make_fixture()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="psr")
    p.add_argument(
        "--fixtures", action="store_true", help="use tests/fixtures instead of real data"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("fetch", help="download the ESCI files")
    s.add_argument("--if-changed", action="store_true")
    s.set_defaults(func=_fetch)

    s = sub.add_parser("prepare", help="filter, join and split the raw files")
    s.set_defaults(func=_prepare)

    s = sub.add_parser("make-fixture", help="sample the test fixture from data/processed")
    s.set_defaults(func=_make_fixture)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)

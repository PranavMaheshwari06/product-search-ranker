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


def _index(args):
    from product_search_ranker.retrieval import bm25

    fixtures = config.fixtures_mode(args.fixtures)
    for locale in config.LOCALES:
        for field in args.fields.split(","):
            bm25.build(locale, field, fixtures)


def _eval(args):
    from product_search_ranker.evaluate.run import evaluate

    systems = [s for s in args.systems.split(",") if s]
    evaluate(systems, args.split, fixtures=config.fixtures_mode(args.fixtures))


def _llm_ping(args):
    from product_search_ranker.llm.client import ping

    ping(args.provider)


def _judge(args):
    from product_search_ranker.judge.judge import run

    fixtures = config.fixtures_mode(args.fixtures)
    run(fixtures, provider=args.provider or config.llm_provider(fixtures), record=args.record)


def _agreement(args):
    from product_search_ranker.judge.agreement import run

    systems = [s for s in args.systems.split(",") if s] if args.systems else None
    run(systems, fixtures=config.fixtures_mode(args.fixtures))


def _qu(args):
    from product_search_ranker.llm.query_understanding import run

    fixtures = config.fixtures_mode(args.fixtures)
    provider = args.provider or config.llm_provider(fixtures)
    run(fixtures, provider=provider, record=args.record, limit=args.limit)


def _add_llm_args(s) -> None:
    s.add_argument(
        "--provider",
        choices=("replay", "gemini", "ollama"),
        help="default: replay with --fixtures, else PSR_LLM_PROVIDER or gemini",
    )
    s.add_argument("--record", action="store_true", help="save real replies as replay fixtures")


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

    s = sub.add_parser("index", help="build the BM25 indexes (skips finished ones)")
    s.add_argument("--fields", default="all,title,bullets,brand,desc")
    s.set_defaults(func=_index)

    s = sub.add_parser("eval", help="score a split and merge the results into metrics.json")
    s.add_argument("--systems", required=True, help="comma-separated, e.g. bm25")
    s.add_argument("--split", default="test", choices=config.SPLITS)
    s.set_defaults(func=_eval)

    s = sub.add_parser("llm-ping", help="one tiny LLM call; picks the Gemini Flash model if unset")
    s.add_argument("--provider", choices=("replay", "gemini", "ollama"))
    s.set_defaults(func=_llm_ping)

    s = sub.add_parser("qu", help="LLM query understanding for hard queries (resumable)")
    _add_llm_args(s)
    s.add_argument("--limit", type=int, help="only the first N uncached queries (a test drive)")
    s.set_defaults(func=_qu)

    s = sub.add_parser("judge", help="LLM-label the judged sample (resumable)")
    _add_llm_args(s)
    s.set_defaults(func=_judge)

    s = sub.add_parser("agreement", help="judge vs human labels -> judge.json, confusion plot")
    s.add_argument("--systems", help="comma-separated; default: every system with a test run")
    s.set_defaults(func=_agreement)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)

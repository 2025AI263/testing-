"""Pipeline runner:  python main.py [extract|transform|load|eda|all]

  extract    fetch raw comments from the YouTube API (needs YOUTUBE_API_KEY, uses quota)
  transform  clean + topic + score the raw file and print a summary (saves nothing)
  load       transform, then write data/comments_clean.csv and the SQLite database
  eda        rebuild data/reports/* from the database
  all        extract -> load -> eda
"""
import sys


def step_extract():
    import extract
    extract.run()


def step_transform():
    import transform
    transform.run()
    print("Dry run only. Use `python main.py load` to save the results.")


def step_load():
    import transform, load
    load.run(transform.run())


def step_eda():
    import eda
    eda.run()


def step_all():
    step_extract()
    step_load()
    step_eda()


STEPS = {"extract": step_extract, "transform": step_transform, "load": step_load,
         "eda": step_eda, "all": step_all}


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step not in STEPS:
        sys.exit(f"Unknown step: {step}\nUsage: python main.py [{'|'.join(STEPS)}]")
    STEPS[step]()

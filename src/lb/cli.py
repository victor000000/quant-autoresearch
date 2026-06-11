"""Console-script entry points (see pyproject [project.scripts])."""


def round_main():
    import runpy
    import os
    from lb.paths import ROOT
    runpy.run_path(os.path.join(ROOT, "scripts", "run_round.py"), run_name="__main__")

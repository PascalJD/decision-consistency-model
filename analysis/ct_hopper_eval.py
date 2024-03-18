if __name__ == '__main__':
    from ml_logger import logger, instr, needs_relaunch
    from analysis import RUN
    import jaynes
    from scripts.ct_hopper_eval import evaluate
    from config.ct_hopper import Config
    from params_proto.neo_hyper import Sweep

    sweep = Sweep(RUN, Config).load("/home/pjutrasd/depot_symlink/projects/consistency_decision/analysis/json/ct_hopper.jsonl")

    for kwargs in sweep:
        logger.print(RUN.prefix, color='green')
        jaynes.config("local")
        thunk = instr(evaluate, **kwargs)
        jaynes.run(thunk)

    jaynes.listen()

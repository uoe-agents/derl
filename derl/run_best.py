import subprocess
import multiprocessing
from itertools import product
from pathlib import Path
import re
import yaml

import click

_CPU_COUNT = multiprocessing.cpu_count() - 1


def config_to_args(algorithm):
    with open(f"best_config/{algorithm}.yaml", "r") as config_file:
        config = yaml.load(config_file)
    main_args = []
    for k, v in config["main"].items():
        main_args.append(f"{k}={v}")
    curiosity_args = {}
    for cur, cur_conf in config["curiosities"].items():
        cur_args = []
        if cur_conf is not None:
            for k, v in cur_conf.items():
                cur_args.append(f"{k}={v}")
        curiosity_args[cur] = cur_args
    return main_args, curiosity_args


def _compute_combinations(algorithm, env, curiosity, seeds):
    click.echo("Found following config: ")
    main_config, curiosity_configs = config_to_args(algorithm)
    main_config.append(f"+env={env}")

    if curiosity not in curiosity_configs:
        valid_curiosities = [f"'{cur}'" for cur in list(curiosity_configs.keys())]
        click.echo(
            f"Intrinsic reward '{curiosity}' is not defined for config '{algorithm}'. Defined intrinsic rewards are {', '.join(valid_curiosities)}."
        )
        raise click.Abort()
    curiosity_config = curiosity_configs[curiosity]
    if curiosity_config is not None:
        config = main_config + curiosity_config
    click.echo(config)
    configs = []
    for seed in range(seeds):
        configs.append(config + [f"seed={seed}"])
    return configs


def work(cmd):
    cmd = cmd.split(" ")
    return subprocess.call(cmd, shell=False)


@click.group()
def cli():
    pass


@cli.group()
@click.option("--seeds", default=1, show_default=True, help="How many seeds to run")
@click.argument("env")
@click.argument("algorithm")
@click.argument("curiosity")
@click.pass_context
def run(ctx, algorithm, env, curiosity, seeds):
    combos = _compute_combinations(algorithm, env, curiosity, seeds)
    if len(combos) == 0:
        click.echo("No valid combinations. Aborted!")
        raise click.Abort()
    ctx.obj = combos


@run.command()
@click.option(
    "--cpus",
    default=_CPU_COUNT,
    show_default=True,
    help="How many processes to run in parallel",
)
@click.pass_obj
def start(combos, cpus):
    configs = ["python run.py " + " ".join([c for c in combo if c.startswith("--")]) + " ".join([c for c in combo if not c.startswith("--")]) for combo in combos]

    click.confirm(
        f"Chosen configuration will be run for {len(configs)} seeds. {min(cpus, len(configs))} will run in parallel. Continue?",
        abort=True,
    )

    pool = multiprocessing.Pool(processes=cpus)
    print(pool.map(work, configs))


if __name__ == "__main__":
    cli()

from inspect import Parameter, signature

import torch

def try_get_function(module, name):
    try:
        func = getattr(module, name)
    except AttributeError:
        print(f"{name} is not a valid name in the {module} module.")
        raise
    return func

def check_scheduler_args(func, name, args):

    for pname, p in signature(func).parameters.items():
        if pname != "optimizer" and p.default == Parameter.empty:
            assert (
                pname in args.keys()
            ), f"{name} expects a value for {pname}"

def get_optimizer(model, name, args):
    opt = try_get_function(torch.optim, name)
    return opt(model.parameters(), **args)

def get_scheduler(optimizer, name, args):

    scheduler = try_get_function(torch.optim.lr_scheduler, name)
    check_scheduler_args(scheduler, name, args)
    return scheduler(optimizer, **args)

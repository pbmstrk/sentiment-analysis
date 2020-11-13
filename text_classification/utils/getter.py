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

def get_optimizer(model, name, args, no_decay=("bias", "LayerNorm.weight")):
    opt = try_get_function(torch.optim, name)
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.get("weight_decay", 0.0),
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    args.pop("weight_decay", None)
    return opt(optimizer_grouped_parameters, **args)

def get_scheduler(optimizer, name, args):

    scheduler = try_get_function(torch.optim.lr_scheduler, name)
    check_scheduler_args(scheduler, name, args)
    return scheduler(optimizer, **args)

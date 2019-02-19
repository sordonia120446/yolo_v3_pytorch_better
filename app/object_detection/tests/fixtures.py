import torch


def _train_step(model, loss_fn, optim, batch):
    """Run a training step on model for a given batch of data
    Parameters of the model accumulate gradients and the optimizer performs
    a gradient update on the parameters
    Parameters
    ----------
    model : torch.nn.Module
        torch model, an instance of torch.nn.Module
    loss_fn : function
        a loss function from torch.nn.functional
    optim : torch.optim.Optimizer
        an optimizer instance
    batch : list
        a 2 element list of inputs and labels, to be fed to the model
    """

    # put model in train mode
    model.train()
    if torch.cuda.is_available():
        model.cuda()

    # run one forward + backward step
    # clear gradient
    optim.zero_grad()

    # inputs and targets
    inputs, targets = batch[0], batch[1]

    # move data to GPU
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        targets = targets.cuda()
    # forward
    likelihood = model(inputs)
    # calc loss
    loss = loss_fn(likelihood, targets)
    # backward
    loss.backward()
    # optimization step
    optim.step()


def var_change_helper(model, loss_fn, optim, batch, params=[]):
    """Check if given variables (params) change or not during training
    If parameters (params) aren't provided, check all parameters.
    Parameters
    ----------
    model : torch.nn.Module
        torch model, an instance of torch.nn.Module
    loss_fn : function
        a loss function from torch.nn.functional
    optim : torch.optim.Optimizer
        an optimizer instance
    batch : list
        a 2 element list of inputs and labels, to be fed to the model
    params : list, optional
        list of parameters of form (name, variable)
    Raises
    ------
    VariablesChangeException
        if vars_change is True and params DO NOT change during training
        if vars_change is False and params DO change during training
    """
    if not params:
        # get a list of params that are allowed to change
        for named_param in model.named_parameters():
            if named_param[1].requires_grad:
                params.append(named_param)

    # take a copy
    initial_params = [(name, p.clone()) for (name, p) in params]

    # run a training step
    _train_step(model, loss_fn, optim, batch)

    # check if variables have changed
    return initial_params, params

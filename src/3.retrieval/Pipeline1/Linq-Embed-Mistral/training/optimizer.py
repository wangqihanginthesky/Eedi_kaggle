import torch

class OptimizerFacade(object):
    @staticmethod
    def create_optimizer(optimizer_config, model):
        # Initialize optimizer
        optimizer_name = optimizer_config.name

        if optimizer_name == 'AdamW':
            optimizer = OptimizerFacade.create_AdamW(optimizer_config, model)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        return optimizer

    @staticmethod
    def create_AdamW(optimizer_config, model):
        optimizer_params = optimizer_config.params
        optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer_params['lr'], weight_decay=optimizer_params['weight_decay'])
        return optimizer




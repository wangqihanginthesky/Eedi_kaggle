from transformers import get_cosine_schedule_with_warmup

class SchedulerFacade(object):
    @staticmethod
    def create_scheduler(lr_scheduler_config, optimizer, train_step_size):
        # Initialize scheduler
        scheduler_name = lr_scheduler_config.name

        if scheduler_name == 'cosine_schedule_with_warmup':
            scheduler = SchedulerFacade.create_cosine_schedule_with_warmup(lr_scheduler_config, optimizer, train_step_size)
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
        return scheduler

    @staticmethod
    def create_cosine_schedule_with_warmup(lr_scheduler_config, optimizer, train_step_size):
        interval = lr_scheduler_config.interval
        if interval=='step':
            scheduler_params = lr_scheduler_config.params
            max_epochs = scheduler_params['max_epochs']
            warmup_steps_ratio = scheduler_params['warmup_steps_ratio']
            
            total_steps = train_step_size * max_epochs
            warmup_steps = int(total_steps * warmup_steps_ratio)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                num_cycles=scheduler_params['num_cycles']
            )
        else:
            raise ValueError(f"Unsupported interval: {interval}")
        return scheduler
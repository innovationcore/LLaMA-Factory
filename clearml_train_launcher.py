import argparse
import uuid

from clearml import Task

def init_clone_task():

    template_task = Task.get_task(project_name=args.base_project_name, task_name=args.base_task_name)
    template_project_id = template_task.get_project_id(args.base_project_name)
    print('Template project_id:', template_project_id)

    trainer_task_name = 'llm_factory_trainer-' + str(uuid.uuid4())
    print('Creating new task:', trainer_task_name)
    cloned_task = Task.clone(
        source_task=template_task,
        name=trainer_task_name,
        comment='automatically created task based on a template',
        project=template_project_id,
    )

    return trainer_task_name, cloned_task

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LLM Factory Agent')

    # general args
    parser.add_argument('--base_project_name', type=str, default='llm_factory_trainer', help='name of project')
    parser.add_argument('--base_task_name', type=str, default='trainer_template_v0_campus_A100', help='name of project')
    parser.add_argument('--queue_name', type=str, default='campus_A100_llm', help='name of project')

    # get args
    args = parser.parse_args()

    print('Get ClearML Template:', 'project_name:', args.base_project_name, 'task_name:', args.base_task_name)
    template_task = Task.get_task(project_name=args.base_project_name, task_name=args.base_task_name)
    project_id = template_task.get_project_id(args.base_project_name)
    print('Template project_id:', project_id)

    trainer_task_name = 'llm_factory_trainer-' + str(uuid.uuid4())
    print('Creating new task:', trainer_task_name)
    cloned_task = Task.clone(
        source_task=template_task,
        name=trainer_task_name,
        comment='automatically created task based on a template',
        project=project_id,
    )
    
    # Set parameters (replaces existing hyperparameters in task)
    parameters = template_task.get_parameters(args.base_project_name)

    parameters['Args/stage'] = 'pt'
    parameters['Args/dataset'] = 'generic_text'
    cloned_task.set_parameters(parameters)

    Task.enqueue(
        task=cloned_task,
        queue_name=args.queue_name,
        queue_id=None
    )


import argparse
import json
import uuid

from clearml import Task

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLM Factory Agent')

    # general args
    parser.add_argument('--base_project_name', type=str, default='llm_factory_trainer', help='name of project')
    parser.add_argument('--base_task_name', type=str, default='trainer_template_v0', help='name of project')
    parser.add_argument('--queue_name', type=str, default='test_queue', help='name of project')

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

    # get parameters from template
    parameters = template_task.get_parameters(args.base_project_name)

    parameters['Args/dataset_project'] = 'datasets'
    parameters['Args/dataset_name'] = 'example_generic_instruct.json'
    parameters['Args/dataset_file'] = 'example_generic_instruct.json'

    # set new params
    cloned_task.set_parameters(parameters)

    Task.enqueue(
        task=cloned_task,
        queue_name=args.queue_name,
        queue_id=None
    )

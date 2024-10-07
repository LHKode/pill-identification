from typing import List
from pills_identification.workflows import IWorkflow, IWorkflowStep


class PillsWorkflow(IWorkflow):
    pre_workflows: List[IWorkflow]
    steps: List[IWorkflowStep]
    pos_workflows: List[IWorkflow]

    def __init__(
        self,
        steps: List[IWorkflowStep],
        pre_workflows: List[IWorkflow] = None,
        pos_workflows: List[IWorkflow] = None,
        **kwargs,
    ) -> None:
        if pre_workflows is None:
            pre_workflows = []
        if pos_workflows is None:
            pos_workflows = []

        super().__init__(**kwargs)
        self.steps = steps
        self.pre_workflows = pre_workflows
        self.pos_workflows = pos_workflows

    def __call__(self, **kwargs):
        result = super().__call__(**kwargs)

        if len(self.pre_workflows) > 0:
            for workflow in self.pre_workflows:
                result = workflow.__call__(**result)

        if len(self.steps) > 0:
            for step in self.steps:
                result = step.__call__(**result)

        if len(self.pos_workflows) > 0:
            for workflow in self.pos_workflows:
                result = workflow.__call__(**result)

        return result


class PillsWorkflowStep(IWorkflowStep):
    pass

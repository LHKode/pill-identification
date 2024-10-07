from pills_identification.workflows.pills_workflow import PillsWorkflowStep


class DummyAlignmentStep(PillsWorkflowStep):
    def __call__(self, **kwargs):
        print(DummyAlignmentStep.__name__ + "receive" + str(kwargs))
        return super().__call__(**kwargs)

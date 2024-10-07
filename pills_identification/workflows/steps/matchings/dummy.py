from pills_identification.workflows.pills_workflow import PillsWorkflowStep


class DummyMatchingStep(PillsWorkflowStep):
    def __call__(self, **kwargs):
        print(DummyMatchingStep.__name__ + "receive" + str(kwargs))
        return super().__call__(**kwargs)

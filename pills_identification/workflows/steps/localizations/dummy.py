from pills_identification.workflows.pills_workflow import PillsWorkflowStep


class DummyLocalizationStep(PillsWorkflowStep):
    def __call__(self, **kwargs):
        print(DummyLocalizationStep.__name__ + "receive" + str(kwargs))
        return super().__call__(**kwargs)

from pills_identification.workflows.pills_workflow import PillsWorkflowStep


class DummyEmbeddingStep(PillsWorkflowStep):
    def __call__(self, **kwargs):
        print(DummyEmbeddingStep.__name__ + "receive" + str(kwargs))
        return super().__call__(**kwargs)

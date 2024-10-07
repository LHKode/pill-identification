from pills_identification.workflows.pills_workflow import PillsWorkflow

from pills_identification.workflows.steps.localizations.dummy import DummyLocalizationStep
from pills_identification.workflows.steps.alignments.dummy import DummyAlignmentStep
from pills_identification.workflows.steps.embeddings.dummy import DummyEmbeddingStep
from pills_identification.workflows.steps.matchings.dummy import DummyMatchingStep


def main():
    steps = [
        DummyLocalizationStep(),
        DummyAlignmentStep(),
        DummyEmbeddingStep(),
        DummyMatchingStep(),
    ]

    workflow = PillsWorkflow(steps=steps)
    workflow(image_dir="../images")


if __name__ == "__main__":
    main()

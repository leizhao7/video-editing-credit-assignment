# Training runtime draft

Fixed-PRM source, a separate decay overlay, provenance hashes and synthetic CPU tests are included here. See [the training guide](../docs/training.md) for the code map, exact decay schedule, test commands and build limitations.

Start with the [CPU setup and test commands](../docs/training.md#run-the-offline-cpu-tests). The fixed and decay suites pass in a fresh Python 3.12 CPU environment. These checks do not launch training.

A portable GPU training build and public cloud launcher remain pending. Newly generated tasks also need a [trainer preparation adapter](../docs/training.md#using-your-own-tasks); changing dataset paths is not sufficient. Retained historical source guards must not be bypassed.

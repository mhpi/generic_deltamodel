# Contributing to *𝛿MG*

Thank you for considering contributing to this project! Whether it's fixing a bug, improving documentation, or adding a new feature, we welcome contributions.

There is a minimal set of standards we would ask you to consider to speed up the review process.

## 🧭 How to Contribute

1. **Fork the repository**
   - If you have not already done so, create a fork of the `generic_deltamodel` repo (master branch) and make changes to this copy.

2. **Lint & test your code**
   - Make sure development packages for 𝛿MG are installed. This can be done by flagging dev packages during pip install like `uv pip install "./generic_deltamodel[dev]` (see also: [setup](./setup.md)).

   - Once your changes are complete, run the following in your Python environement:

      ```bash
      cd ./generic_deltamodel

      pytest tests

      pre-commit install

      git add .

      git commit -m 'your commit message'
      ```

     Upon committing, pre-commit will run a series of checks according to `.pre-commit-config.yaml` lint and format your code. This will block your commit if changes are made or requested. If manual changes are required, you will be notified. If only automatic changes are made, simply perform the git add and commit once more to push your code.

     Note: if pytest does not work, try `python -m pytest`.

   - If ruff or pytest report any errors, please try to correct these if possible. Otherwise, do `pre-commit uninstall` to proceed with committing your code and we can help in the next step.

3. **Make a pull request (PR)**
    - When you are ready, make a PR of your fork to the `generic_deltamodel` repository master branch.

    - In the PR description, include enough detail so that we can understand the changes made and rationale if necessary.

    - If the `generic_deltamodel` master branch has new commits not included in your forked version, we would ask you to merge these new changes into your fork before we accept the PR. We can assist with this if necessary.

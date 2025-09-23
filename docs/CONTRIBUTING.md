# Contributing to *dMG*

Thank you for considering contributing to this project! Whether it's fixing a bug, improving documentation, or adding a new feature, we welcome contributions.

We have a minimal set of standards we would ask you to consider to speed up the review process.

## 🧭 How to Contribute

1. **Fork the repository**
   - If you have not already done so, create a fork of the dMG repo (master branch) and make changes to this copy.

2. **Lint & test your code**
   - If you have not already, install development packages for dMG (see [setup](./setup.md)).

   - When your changes are ready, activate the dMG ENV and run

      ```bash
      cd ./generic_deltamodel
      
      isort .
      
      ruff check .

      pytest tests
      ```

   - If there ruff or pytest report any errors, please try to correct these if possible. (We can also help in the next step).

3. **Make a pull request (PR)**
    - When you are ready make a PR to the dMG repository master branch.

    - In the PR description, make sure to include enough detail so that we can understand the changes made and rationale if necessary.

    - If dMG master branch has new commits not included in your forked version, we would ask you to merge these new changes into your fork before we accept the PR. We can assist with this if necessary.

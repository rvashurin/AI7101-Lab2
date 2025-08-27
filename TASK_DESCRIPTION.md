# Lab 2 Task Description

This lab is focused on switching branches in git, conflict resolution when syncing feature branches with main branch, working with forks, and creating pull requests. It is based on the reference solution to the Lab 1.

The original repo (this one) contains a new feature branch `add_third_poly` that adds a 3rd degree polynomial regression experiment. Unfortunately, someone else was working on another feature - expanding grid of the parameters for linear regression. This feature was merged in [another pull request](https://github.com/rvashurin/AI7101-Lab2/pull/1) before new experiment was added. When [pull request](https://github.com/rvashurin/AI7101-Lab2/pull/2) was opened for the polynomial update, merge conflict emerged with the main branch.

Your task is to fix these conflicts and create a new pull request that does not conflict with the main branch.

You should:

1. Fork the original repo. This will create a copy of the repo under you GitHub Account. Use the "Fork" button on the top right of the page.
2. Clone the repo to your machine (`git clone %your fork url%`). URL can be copied from the "Code" button on the main page of your fork.
3. Fetch the feature branch (`git fetch origin add_third_poly`).
4. Sync it with main branch using either `git merge main` or `git rebase main`.
5. Resolve all the conflicts that arise during the process so that both new feature and updates made to the main branch coexist nicely.
6. Create a pull request from the feature branch **in your fork** to the main branch of the **original repo**. To do that:
	1. Go to the original repo, click on Pull Requests tab and click on New pull request button.
	2. Click on compare across forks link at the top of the page.
	3. Select your fork in the `head repository` dropdown.
	4. And select an updated feature branch in the adjacent dropdown.
7. Name of the pull request should start with you name (i.e. JonSmith/3rd Degree Poly). 

Your submission should be the number of the pull request you have created to the original repo. Your pull request should not contain any new commits except merge commit if you used `git merge` to reconcile branches. It should be ready to merge - no conflicts with main branch should remain.

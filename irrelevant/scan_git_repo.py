"""
Yuli Tshuva
Find all notebooks in the git repo.
"""

from os.path import join, isdir
from os import listdir


def find_notebooks_in_git_repo(repo_path):
    for file in listdir(repo_path):
        if file.endswith(".ipynb"):
            print(join(repo_path, file))
        if isdir(join(repo_path, file)):
            find_notebooks_in_git_repo(join(repo_path, file))


find_notebooks_in_git_repo("tesa_ordinal_similarity")

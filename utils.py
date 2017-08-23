
# Standard libraries
import subprocess

def get_version_from_tag():
    cmd = "git describe --tags --dirty"
    git_description = subprocess.check_output(cmd.split())
    # Will return the closest tag: e.g. 0.1.0
    # then add -X-SHA-1 where X is the number of commit
    # between HEAD and that tag, SHA--1 is the current head SHA-1
    # e.g: 0.1.0-10-g48b85
    # If HEAD is on the tag, nothing is added
    # Finally, it adds -dirty if changes are not commited
    # e.g: 0.0.1-10-g48b85f5-dirty

    git_description = git_description.split('\n')[0].split("-")
    version = git_description[0]
    if len(git_description) >= 3:
        # There is a number of commits and a SHA-1
        version += "-" + git_description[1]
    if git_description[-1] == "dirty":
        # The project is modified
        version += "-dev"
    return version

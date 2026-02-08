import os
import pytest

# We just want to check if the input folder exists for now
def test_folders_exist():
    assert os.path.exists("input_images") == True
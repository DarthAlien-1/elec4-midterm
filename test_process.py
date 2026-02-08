import os
import pytest

def test_folders_exist():
    assert os.path.exists("input_images") == True

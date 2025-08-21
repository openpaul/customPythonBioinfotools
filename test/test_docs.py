import pathlib
import pytest

from mktestdocs import check_md_file


# Test all markdown files in docs/ directory
@pytest.mark.parametrize("fpath", pathlib.Path("docs").glob("**/*.md"), ids=str)
def test_docs_examples(fpath):
    """Test that all Python code blocks in documentation work correctly."""
    check_md_file(
        fpath=fpath, memory=True
    )  # memory=True allows code blocks to depend on each other

# Development

## Releasing a new version on PyPI.

Currently I have a GitHub workflow to automatically publish to PyPI when a new
[tagged] release is made on GitHub, so the only step required is to make a new
release.

Steps in release process are:

1. Double check that tests pass
2. Double check linting (`pre-commit run --all`)
3. Push an updated CHANGELOG with link to new version you are about to release.
  Make sure the links at the bottom of the document are updated appropriately!
4. Publish GitHub release and ensure that the workflows pass.

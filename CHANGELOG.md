# Changelog

All notable changes to the project are documented in this file.

The format follows [Common Changelog](https://common-changelog.org/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Sanitize molecules from GraphGA ([#5](https://github.com/AustinT/mol_ga/pull/5)) ([@austint])
- GraphGA molecule generation now independent of batch size ([#5](https://github.com/AustinT/mol_ga/pull/5)) ([@austint])

### Added

### Fixed

- Fix import error for python<3.8 ([#3](https://github.com/AustinT/mol_ga/pull/3)) ([@austint])
- Fix unintended use of system random in sampling ([#4](https://github.com/AustinT/mol_ga/pull/4)) ([@austint])
- Fix occasional crash from rdkit errors during crossover ([#5](https://github.com/AustinT/mol_ga/pull/5)) ([@austint])

## [0.1.0] - 2023-09-05

:seedling: Initial public release.

[Unreleased]: https://github.com/AustinT/mol_ga/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/AustinT/mol_ga/releases/tag/v0.1.0

[@austint]: https://github.com/AustinT

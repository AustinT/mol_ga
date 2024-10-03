# Changelog

All notable changes to the project are documented in this file.

The format follows [Common Changelog](https://common-changelog.org/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

### Added

### Fixed

## [0.1.3]

### Fixed

- Checked for empty population at input ([#13](https://github.com/AustinT/mol_ga/pull/13)) ([@austint])

## [0.1.2]

### Fixed

- Fix error in quantile sampling if NaNs are present ([#8](https://github.com/AustinT/mol_ga/pull/8)) ([@austint])

## [0.1.1]

### Changed

- Sanitize molecules from GraphGA ([#5](https://github.com/AustinT/mol_ga/pull/5)) ([@austint])
- GraphGA molecule generation now independent of batch size ([#5](https://github.com/AustinT/mol_ga/pull/5)) ([@austint])

### Fixed

- Fix import error for python<3.8 ([#3](https://github.com/AustinT/mol_ga/pull/3)) ([@austint])
- Fix unintended use of system random in sampling ([#4](https://github.com/AustinT/mol_ga/pull/4)) ([@austint])
- Fix occasional crash from rdkit errors during crossover ([#5](https://github.com/AustinT/mol_ga/pull/5)) ([@austint])

## [0.1.0] - 2023-09-05

:seedling: Initial public release.

[Unreleased]: https://github.com/AustinT/mol_ga/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/AustinT/mol_ga/releases/tag/v0.1.3
[0.1.2]: https://github.com/AustinT/mol_ga/releases/tag/v0.1.2
[0.1.1]: https://github.com/AustinT/mol_ga/releases/tag/v0.1.1
[0.1.0]: https://github.com/AustinT/mol_ga/releases/tag/v0.1.0

[@austint]: https://github.com/AustinT

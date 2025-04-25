.PHONY: test rust-test pytest lint

# Full test suite
test: rust-test pytest

# Rust test suite
rust-test:
	(cd hashable; cargo test)
	(cd smoot-rs; cargo test)
	# pyo3 tests
	cargo test

# Python test suite
pytest:
	maturin develop
	pytest

lint: rust-lint python-lint

rust-lint:
	(cd hashable; cargo fmt; cargo clippy; cargo machete)
	(cd smoot-rs; cargo fmt; cargo clippy; cargo machete)
	# pyo3 lint
	cargo fmt; cargo clippy; cargo machete

python-lint:
	ruff format smoot tests
	ruff check smoot tests --fix

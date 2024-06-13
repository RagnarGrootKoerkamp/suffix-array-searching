run *args='':
    @cargo run -r --quiet -- {{args}}
build:
    @cargo build -r --quiet
record: build
    perf record cargo run -r --quiet
report:
    @perf report
perf: record report
flame: build
    @cargo flamegraph --open

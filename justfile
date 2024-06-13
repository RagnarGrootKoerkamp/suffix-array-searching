run *args='':
    @cargo run -r --quiet -- {{args}}
build:
    @cargo build -r --quiet
record *args='': build
    @perf record ./target/release/sa-layout {{args}}
report:
    @perf report
perf: record report
flame: build
    @cargo flamegraph --open

stat *args='': build
    perf stat -d ./target/release/sa-layout {{args}}

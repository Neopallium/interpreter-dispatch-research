#!/bin/dash
TEST=$1

if [ "$TEST" = "" ]; then
	echo "Usage $0: <path to test binary ./target/release/deps/interpreter_dispatch_research-....>"
	echo "Run 'cargo test --release' to get the test binary filename."
	exit
fi

test() {
	time -f "%e: Test $1" $TEST $1 >/dev/null
}

# Run tests and sort by time.
(
test closure_tree::counter_loop
test switch::counter_loop
test closure_block::counter_loop
test closure_tail_2::counter_loop
test fused::ct3::counter_loop
test switch_2::counter_loop
test fused::rt3::counter_loop
test fused::ct2::counter_loop
test fused::rt2::counter_loop
test enum_tree_2::counter_loop
test switch::more_comps
test fused::rt::counter_loop
test switch_tail::counter_loop
test closure_loop::counter_loop
test switch_tail_2::counter_loop
test enum_tree::counter_loop
test closure_block::more_comps
test closure_tail::counter_loop
test fused::ct::counter_loop
test switch_tail::more_comps
) 2>&1 | sort -n


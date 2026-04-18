/**
 * Phase 1 stub: adapter symmetry assertions. After Phase 3 lands, this
 * file compares the method surfaces of PyodideAdapter and HttpAdapter
 * and asserts they expose the same public API (HTTP may throw
 * NotSupported for interactive primitives — that's still part of the
 * contract). For now the tests are all todo; presence of the file keeps
 * the Phase 3 work visible in the test matrix.
 */
const { test, describe } = require("node:test");

describe("zoomy_cli adapter symmetry — Phase 3 target", () => {
    test.todo("PyodideAdapter and HttpAdapter share the same method names");
    test.todo("runCode / extractParams / describeModel are callable on both (HTTP may throw NotSupported)");
    test.todo("submitCase on both adapters returns a promise resolving to a job handle");
    test.todo("cancelJob is symmetric — resolves to {status:'cancelled'} on both");
});

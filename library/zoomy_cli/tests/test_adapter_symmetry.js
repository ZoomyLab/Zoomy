/**
 * Adapter symmetry: PyodideAdapter and HttpAdapter should expose the
 * same method names so the CLI can dispatch without type checks. The
 * HTTP adapter may throw NotSupportedError for methods that aren't
 * meaningful over HTTP — that's part of the contract.
 */
const { test, describe, before } = require("node:test");
const path = require("path");
const url = require("url");

const CLI_DIR = path.resolve(__dirname, "..");

const SHARED_METHODS = [
    "connect", "disconnect",
    "runCode", "extractParams", "describeModel",
    "openHdf5", "writeHdf5Bytes",
    "submitCase", "cancelJob",
    "listRegistry", "health",
];

describe("PyodideAdapter / HttpAdapter symmetry", () => {
    let mod;
    before(async () => {
        mod = await import(url.pathToFileURL(path.join(CLI_DIR, "node.mjs")).href);
    });

    test("both adapters share the SHARED_METHODS set", () => {
        const py = mod.PyodideAdapter.prototype;
        const http = mod.HttpAdapter.prototype;
        for (const m of SHARED_METHODS) {
            if (typeof py[m] !== "function") throw new Error("PyodideAdapter missing " + m);
            if (typeof http[m] !== "function") throw new Error("HttpAdapter missing " + m);
        }
    });

    test("HttpAdapter.runCode throws NotSupportedError", () => {
        const a = new mod.HttpAdapter({ url: "http://example.invalid" });
        try {
            a.runCode();
            throw new Error("expected NotSupportedError");
        } catch (e) {
            if (!(e instanceof mod.NotSupportedError)) throw new Error("unexpected error: " + e);
        }
    });

    test("HttpAdapter.extractParams / describeModel / openHdf5 / writeHdf5Bytes all throw NotSupportedError", () => {
        const a = new mod.HttpAdapter({ url: "http://example.invalid" });
        for (const method of ["extractParams", "describeModel", "openHdf5", "writeHdf5Bytes"]) {
            try {
                a[method]();
                throw new Error("expected NotSupportedError for " + method);
            } catch (e) {
                if (!(e instanceof mod.NotSupportedError)) throw new Error(method + ": unexpected error " + e);
            }
        }
    });

    test("PyodideAdapter.submitCase / cancelJob / listRegistry / health throw NotSupportedError", () => {
        const a = new mod.PyodideAdapter({});   // no worker — we don't use it
        for (const method of ["submitCase", "cancelJob", "listRegistry", "health"]) {
            try {
                a[method]();
                throw new Error("expected NotSupportedError for " + method);
            } catch (e) {
                if (!(e instanceof mod.NotSupportedError)) throw new Error(method + ": unexpected error " + e);
            }
        }
    });

    test("HttpAdapter requires a URL", () => {
        try {
            new mod.HttpAdapter({});
            throw new Error("expected Error for missing url");
        } catch (e) {
            if (!/url/.test(e.message)) throw e;
        }
    });
});

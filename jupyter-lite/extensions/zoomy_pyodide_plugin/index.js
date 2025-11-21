export default {
  id: "zoomy-pyodide-preload",
  autoStart: true,

  activate: async (app, _) => {
    console.log("[zoomy] Preloading packages into Pyodide...");

    // Wait for pyodide to be ready
    await window.pyodideReadyPromise;

    // Pre-install wheels from your embedded pypi index
    const micropip = window.pyodide.pyimport("micropip");

    await micropip.install("zoomy-core");
    await micropip.install("meshio");

    console.log("[zoomy] Packages installed.");
  }
};


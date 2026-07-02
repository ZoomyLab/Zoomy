import argparse
import importlib


def main():
    parser = argparse.ArgumentParser(description="Zoomy Solver Server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--adapter", default="numpy", help="Built-in: numpy, jax. Or full class path: mypackage.MyAdapter")
    args = parser.parse_args()

    builtin = {
        "numpy": "zoomy_server.adapters.numpy.NumpyAdapter",
        "jax": "zoomy_server.adapters.jax.JaxAdapter",
        "amrex": "zoomy_server.adapters.amrex.AmrexAdapter",
        "dmplex": "zoomy_server.adapters.dmplex.DmplexAdapter",
        "firedrake": "zoomy_server.adapters.firedrake.FiredrakeAdapter",
        "foam": "zoomy_server.adapters.foam.FoamAdapter",
        "mesh": "zoomy_server.adapters.mesh.MeshAdapter",
    }

    adapter_path = builtin.get(args.adapter, args.adapter)
    mod_path, cls_name = adapter_path.rsplit(".", 1)
    mod = importlib.import_module(mod_path)
    adapter_cls = getattr(mod, cls_name)

    from zoomy_server.server import ZoomyServer
    server = ZoomyServer(adapter_cls())
    server.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()

import json
import param


def extract_param_schema(class_path, init_kwargs=None):
    module_path, class_name = class_path.rsplit(".", 1)
    mod = __import__(module_path, fromlist=[class_name])
    cls = getattr(mod, class_name)

    skip = {"name"}
    schema = {"class": class_path, "params": {}}

    for pname, p in cls.param.objects("existing").items():
        if pname in skip:
            continue
        ptype = type(p).__name__
        entry = {"type": ptype, "default": None, "doc": p.doc or ""}

        if isinstance(p, param.Integer):
            entry["default"] = p.default if not callable(p.default) else None
            entry["bounds"] = getattr(p, "bounds", None)
            entry["step"] = getattr(p, "step", 1)
        elif isinstance(p, param.Number):
            entry["default"] = p.default if not callable(p.default) else None
            entry["bounds"] = getattr(p, "bounds", None)
            entry["step"] = getattr(p, "step", None)
        elif isinstance(p, param.Boolean):
            entry["default"] = p.default
        elif isinstance(p, param.String):
            entry["default"] = p.default
        elif isinstance(p, (param.Selector, param.ObjectSelector)):
            entry["default"] = p.default
            entry["objects"] = list(p.objects or [])
        elif isinstance(p, param.Parameter):
            val = p.default
            if callable(val):
                entry["default"] = None
                entry["callable"] = True
            elif isinstance(val, dict):
                entry["type"] = "Dict"
                entry["default"] = {}
                for k, v in val.items():
                    if isinstance(v, (list, tuple)) and len(v) == 2:
                        entry["default"][k] = v[0]
                    elif isinstance(v, (int, float)):
                        entry["default"][k] = v
                    else:
                        entry["default"][k] = 0.0
            else:
                entry["default"] = val if isinstance(val, (int, float, str, bool)) else None

        schema["params"][pname] = entry

    return json.dumps(schema)

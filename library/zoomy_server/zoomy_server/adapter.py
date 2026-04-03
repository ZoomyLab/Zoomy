class SolverAdapter:
    tag = "base"

    def solve(self, case, output_dir, on_progress):
        raise NotImplementedError

    def list_models(self):
        return []

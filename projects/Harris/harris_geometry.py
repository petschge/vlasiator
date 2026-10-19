"""Sheet geometry for the Harris/GEM diagnostics, shared so that every script handles both the
fully periodic double sheet and the single sheet between conducting walls.

Everything is read from one (normally the first) output:
  interior    rows with no cell flagged as a boundary in vg_boundarytype (all rows if absent)
  sheets      sign changes of the x-averaged Bx between interior rows, interpolated, with the
              sign of dBx/dy (+1 is the paper's orientation; the double sheet's sheet 2 has -1)
  rows/ypap   per sheet, the interior rows nearer to it than to any other sheet and within its
              half-height, ordered by y in that sheet's frame, ypap = sign*(y - y_sheet)
  half        distance from the sheet to the nearest wall face (walls) or symmetry plane
  lobe_rows   two rows far from every sheet for B0: midway between the double sheets, or the
              two interior rows next to the upper wall of a single sheet
"""
import numpy as np

NOT_SYSBOUNDARY = 1    # sysboundarytype in common.h


class Geometry:
    def __init__(self, f, nx, ny):
        xmin, ymin, _, xmax, ymax, _ = f.get_spatial_mesh_extent()
        self.nx, self.ny = nx, ny
        self.dx, self.dy = (xmax - xmin) / nx, (ymax - ymin) / ny
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        self.xc = xmin + (np.arange(nx) + 0.5) * self.dx
        self.yc = ymin + (np.arange(ny) + 0.5) * self.dy
        order = np.argsort(f.read_variable("CellID"))
        def grid(name):
            a = f.read_variable(name)[order]
            return a.reshape((ny, nx) + a.shape[1:])     # CellID-1 = i + nx*j
        if f.check_variable("vg_boundarytype"):
            bt = grid("vg_boundarytype")
            self.interior = np.all(bt == NOT_SYSBOUNDARY, axis=1)
        else:
            self.interior = np.ones(ny, bool)
        self.walls = not self.interior.all()
        jin = np.where(self.interior)[0]
        self.j_lo, self.j_hi = jin[0], jin[-1]
        self.y_lo = ymin + self.j_lo * self.dy            # wall faces (domain edges if periodic)
        self.y_hi = ymin + (self.j_hi + 1) * self.dy
        bx = grid("vg_b_vol")[..., 0].mean(axis=1)
        self.sheets = []
        for j in jin[:-1]:
            if self.interior[j + 1] and np.sign(bx[j]) != np.sign(bx[j + 1]):
                ys = self.yc[j] + bx[j] * self.dy / (bx[j] - bx[j + 1])
                self.sheets.append((ys, int(np.sign(bx[j + 1] - bx[j]))))
        if not self.sheets:
            raise SystemExit("harris_geometry: no Bx reversal found in the first output")
        Ly = ymax - ymin
        def dist(y, ys):
            d = y - ys
            return d if self.walls else (d + 0.5 * Ly) % Ly - 0.5 * Ly
        self.half, self.rows, self.ypap = [], [], []
        for k, (ys, sg) in enumerate(self.sheets):
            if self.walls:
                h = min(ys - self.y_lo, self.y_hi - ys)
            else:
                others = [abs(dist(o, ys)) for i, (o, _) in enumerate(self.sheets) if i != k]
                h = 0.5 * min(others) if others else 0.5 * Ly
            d = np.array([dist(y, ys) for y in self.yc])
            near = np.array([min(range(len(self.sheets)), key=lambda i: abs(dist(y, self.sheets[i][0])))
                             for y in self.yc])
            sel = np.where(self.interior & (near == k) & (np.abs(d) <= h + 1e-9 * self.dy))[0]
            yp = sg * d[sel]
            o = np.argsort(yp)
            self.half.append(h)
            self.rows.append(sel[o])
            self.ypap.append(yp[o])
        if self.walls:
            self.lobe_rows = (self.j_hi - 1, self.j_hi)
        else:
            ymid = 0.5 * (ymin + ymax) if len(self.sheets) == 2 else self.sheets[0][0] + 0.5 * Ly
            self.lobe_rows = self.straddle(ymid)

    def straddle(self, y):
        """The two cell rows straddling y (sheet centres sit on cell faces)."""
        j0 = int(np.floor((y - self.ymin) / self.dy - 0.5)) % self.ny
        return j0, (j0 + 1) % self.ny

    def _offsets(self, y):
        Ly = self.ymax - self.ymin
        return [y - ys if self.walls else (y - ys + 0.5 * Ly) % Ly - 0.5 * Ly for ys, _ in self.sheets]

    def offset_from_nearest(self, y):
        """Signed offset of y from the nearest sheet centre, in the simulation frame."""
        return min(self._offsets(y), key=abs)

    def nearest_sheet(self, y):
        """Index of the sheet nearest to y."""
        return int(np.argmin(np.abs(self._offsets(y))))

    def describe(self):
        s = ", ".join(f"y={ys:+.4e} m (dBx/dy {'+' if sg > 0 else '-'})" for ys, sg in self.sheets)
        w = (f"walls at y={self.y_lo:+.4e}, {self.y_hi:+.4e} m ({self.ny - self.interior.sum()} boundary rows)"
             if self.walls else "periodic in y")
        return f"{len(self.sheets)} sheet(s): {s}; {w}"

    def flux_function(self, Bx, By):
        """A_z with Bx = dA/dy, By = -dA/dx on the interior rows (NaN elsewhere): integrated along
        x on the row nearest the first sheet, then along y in every column (trapezoidal), so it
        needs no periodicity in y. Any net By flux along that row (nonzero only through div B
        error) is removed as a linear trend to keep A periodic in x."""
        j0 = self.straddle(self.sheets[0][0])[0]
        a0 = -np.concatenate(([0.0], np.cumsum(0.5 * (By[j0, :-1] + By[j0, 1:])) * self.dx))
        drift = -(0.5 * (By[j0, -1] + By[j0, 0])) * self.dx + a0[-1]  # A after one full period
        a0 = a0 - drift * np.arange(self.nx) / self.nx
        A = np.full(Bx.shape, np.nan)
        A[j0] = a0
        for j in range(j0 + 1, self.j_hi + 1):
            A[j] = A[j - 1] + 0.5 * (Bx[j - 1] + Bx[j]) * self.dy
        for j in range(j0 - 1, self.j_lo - 1, -1):
            A[j] = A[j + 1] - 0.5 * (Bx[j + 1] + Bx[j]) * self.dy
        if not self.walls:                        # periodic: rows below j_lo wrap around
            A[~self.interior] = np.nan
        return A

import { useState, useCallback } from "react";

const COLORS = {
  bg: "#0f172a",
  grid: "#1e293b",
  axis: "#475569",
  text: "#e2e8f0",
  textMuted: "#94a3b8",
  macroA: "#3b82f6",
  macroB: "#f59e0b",
  macroC: "#10b981",
  borderL: "rgba(239, 68, 68, 0.7)",
  borderR: "rgba(59, 130, 246, 0.7)",
  borderA: "rgba(245, 158, 11, 0.7)",
  borderB: "rgba(16, 185, 129, 0.7)",
  fillL: "rgba(239, 68, 68, 0.18)",
  fillR: "rgba(59, 130, 246, 0.18)",
  fillA: "rgba(245, 158, 11, 0.18)",
  fillB: "rgba(16, 185, 129, 0.18)",
};

const REL_COLORS = { L: { fill: COLORS.fillL, border: COLORS.borderL }, R: { fill: COLORS.fillR, border: COLORS.borderR }, A: { fill: COLORS.fillA, border: COLORS.borderA }, B: { fill: COLORS.fillB, border: COLORS.borderB } };
const REL_LABELS = { L: "i left of k", R: "i right of k", A: "i above k", B: "i below k" };

// Utility: clip a convex polygon to a half-plane. Returns clipped polygon vertices.
function clipPolygon(poly, testFn) {
  if (poly.length === 0) return [];
  const out = [];
  for (let i = 0; i < poly.length; i++) {
    const cur = poly[i];
    const nxt = poly[(i + 1) % poly.length];
    const curIn = testFn(cur);
    const nxtIn = testFn(nxt);
    if (curIn) out.push(cur);
    if (curIn !== nxtIn) {
      // find intersection
      const dx = nxt[0] - cur[0], dy = nxt[1] - cur[1];
      // binary search for intersection (simple and robust)
      let lo = 0, hi = 1;
      for (let j = 0; j < 30; j++) {
        const mid = (lo + hi) / 2;
        const px = cur[0] + dx * mid, py = cur[1] + dy * mid;
        if (testFn([px, py]) === curIn) lo = mid; else hi = mid;
      }
      const t = (lo + hi) / 2;
      out.push([cur[0] + dx * t, cur[1] + dy * t]);
    }
  }
  return out;
}

// ============== Tab 1: 1D Two Macros ==============
function OneDTwoMacros() {
  const [posA, setPosA] = useState(150);
  const [posB, setPosB] = useState(350);
  const [dragging, setDragging] = useState(null);

  const wA = 60, wB = 80;
  const canvasW = 500, barY = 120, barH = 40;
  const leftOfB = posA + wA / 2 <= posB - wB / 2;
  const rightOfB = posB + wB / 2 <= posA - wA / 2;
  const legal = leftOfB || rightOfB;
  const relation = leftOfB ? "L" : rightOfB ? "R" : "overlap";
  const sep = (wA + wB) / 2;

  const handleMouseDown = (which) => (e) => { e.preventDefault(); setDragging(which); };
  const handleMouseMove = (e) => {
    if (!dragging) return;
    const svg = e.currentTarget;
    const rect = svg.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * canvasW;
    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
    if (dragging === "A") setPosA(clamp(x, wA / 2, canvasW - wA / 2));
    if (dragging === "B") setPosB(clamp(x, wB / 2, canvasW - wB / 2));
  };

  // Position space mapping: x_A on horizontal [0,500], x_B on vertical [0,500] (SVG y inverted)
  const S = 400; // SVG size for position space plot
  const toSvg = (xA, xB) => [xA * S / canvasW, S - xB * S / canvasW];

  // Polyhedron L: x_B >= x_A + sep, plus canvas bounds
  // Vertices of L region: bounded by x_A in [wA/2, W-wA/2], x_B in [wB/2, W-wB/2], x_B >= x_A + sep
  const aMin = wA / 2, aMax = canvasW - wA / 2;
  const bMin = wB / 2, bMax = canvasW - wB / 2;

  const polyL = () => {
    // x_B >= x_A + sep within canvas bounds
    let p = [[aMin, bMin], [aMax, bMin], [aMax, bMax], [aMin, bMax]];
    p = clipPolygon(p, ([a, b]) => b >= a + sep);
    return p.map(([a, b]) => toSvg(a, b));
  };

  const polyR = () => {
    let p = [[aMin, bMin], [aMax, bMin], [aMax, bMax], [aMin, bMax]];
    p = clipPolygon(p, ([a, b]) => a >= b + sep);
    return p.map(([a, b]) => toSvg(a, b));
  };

  const ptsL = polyL();
  const ptsR = polyR();

  return (
    <div>
      <p style={{ color: COLORS.textMuted, fontSize: 13, marginBottom: 12, lineHeight: 1.6 }}>
        <strong style={{ color: COLORS.text }}>1D, two macros.</strong> Macro A (blue, width {wA}) and B (amber, width {wB}) on a line.
        The non-overlap constraint creates two polyhedra in position space (x_A, x_B): one where A is left of B, one where A is right.
        Canvas bounds make them finite triangles. Drag the macros above; the green dot tracks your position in the space below.
      </p>

      {/* Physical 1D view */}
      <svg width="100%" viewBox={`0 0 ${canvasW} 180`}
        style={{ background: COLORS.bg, borderRadius: 8, marginBottom: 12, cursor: dragging ? "grabbing" : "default" }}
        onMouseMove={handleMouseMove} onMouseUp={() => setDragging(null)} onMouseLeave={() => setDragging(null)}>
        <line x1={0} y1={barY + barH / 2} x2={canvasW} y2={barY + barH / 2} stroke={COLORS.axis} strokeWidth={1} />
        <rect x={posA - wA / 2} y={barY} width={wA} height={barH} rx={4}
          fill={legal ? COLORS.macroA : "rgba(239,68,68,0.5)"} stroke={legal ? COLORS.macroA : "#ef4444"} strokeWidth={2} opacity={0.8}
          style={{ cursor: "grab" }} onMouseDown={handleMouseDown("A")} />
        <text x={posA} y={barY + barH / 2 + 5} textAnchor="middle" fill="white" fontSize={12} fontWeight="bold" pointerEvents="none">A</text>
        <rect x={posB - wB / 2} y={barY} width={wB} height={barH} rx={4}
          fill={legal ? COLORS.macroB : "rgba(239,68,68,0.5)"} stroke={legal ? COLORS.macroB : "#ef4444"} strokeWidth={2} opacity={0.8}
          style={{ cursor: "grab" }} onMouseDown={handleMouseDown("B")} />
        <text x={posB} y={barY + barH / 2 + 5} textAnchor="middle" fill="white" fontSize={12} fontWeight="bold" pointerEvents="none">B</text>
        <text x={posA} y={barY - 10} textAnchor="middle" fill={COLORS.textMuted} fontSize={11}>x_A={Math.round(posA)}</text>
        <text x={posB} y={barY + barH + 20} textAnchor="middle" fill={COLORS.textMuted} fontSize={11}>x_B={Math.round(posB)}</text>
        <text x={canvasW / 2} y={30} textAnchor="middle" fill={legal ? "#22c55e" : "#ef4444"} fontSize={14} fontWeight="bold">
          {legal ? `Legal: A is ${relation === "L" ? "left of" : "right of"} B` : "Overlap (infeasible)"}
        </text>
      </svg>

      {/* Position space with actual polyhedra */}
      <svg width="100%" viewBox={`0 0 ${S} ${S}`} style={{ background: COLORS.bg, borderRadius: 8 }}>
        <defs>
          <pattern id="grid1d" width={S / 10} height={S / 10} patternUnits="userSpaceOnUse">
            <path d={`M ${S / 10} 0 L 0 0 0 ${S / 10}`} fill="none" stroke={COLORS.grid} strokeWidth={0.5} />
          </pattern>
        </defs>
        <rect width={S} height={S} fill="url(#grid1d)" />

        {/* Polyhedra */}
        {ptsL.length > 2 && <polygon points={ptsL.map(p => p.join(",")).join(" ")} fill={COLORS.fillR} stroke={COLORS.borderR} strokeWidth={2} />}
        {ptsR.length > 2 && <polygon points={ptsR.map(p => p.join(",")).join(" ")} fill={COLORS.fillL} stroke={COLORS.borderL} strokeWidth={2} />}

        {/* Labels inside polyhedra */}
        {ptsL.length > 2 && (() => {
          const cx = ptsL.reduce((s, p) => s + p[0], 0) / ptsL.length;
          const cy = ptsL.reduce((s, p) => s + p[1], 0) / ptsL.length;
          return <text x={cx} y={cy} textAnchor="middle" fill="rgba(59,130,246,0.9)" fontSize={12} fontWeight="bold">P_L</text>;
        })()}
        {ptsR.length > 2 && (() => {
          const cx = ptsR.reduce((s, p) => s + p[0], 0) / ptsR.length;
          const cy = ptsR.reduce((s, p) => s + p[1], 0) / ptsR.length;
          return <text x={cx} y={cy} textAnchor="middle" fill="rgba(239,68,68,0.9)" fontSize={12} fontWeight="bold">P_R</text>;
        })()}

        {/* Infeasible label */}
        <text x={S / 2} y={S / 2} textAnchor="middle" fill="rgba(239,68,68,0.4)" fontSize={11} fontStyle="italic">infeasible</text>

        {/* Axes labels */}
        <text x={S / 2} y={S - 5} textAnchor="middle" fill={COLORS.textMuted} fontSize={11}>x_A</text>
        <text x={10} y={S / 2} fill={COLORS.textMuted} fontSize={11} transform={`rotate(-90, 10, ${S / 2})`}>x_B</text>

        {/* Current position */}
        {(() => { const [sx, sy] = toSvg(posA, posB); return (
          <g>
            <circle cx={sx} cy={sy} r={6} fill={legal ? "#22c55e" : "#ef4444"} stroke="white" strokeWidth={2} />
            <text x={sx + 10} y={sy - 6} fill="white" fontSize={10}>({Math.round(posA)}, {Math.round(posB)})</text>
          </g>
        ); })()}

        <text x={S / 2} y={16} textAnchor="middle" fill={COLORS.text} fontSize={13} fontWeight="bold">
          Position Space: Two Polyhedra
        </text>
      </svg>
    </div>
  );
}

// ============== Tab 2: 2D Two Macros - 4 Polyhedra in relative coords ==============
function TwoDTwoMacros() {
  const [activeRel, setActiveRel] = useState(null); // null = show all, or L/R/A/B
  const [kx, setKx] = useState(200);
  const [ky, setKy] = useState(200);

  const canvasSize = 400; // chip canvas for this example
  const wI = 50, hI = 35, wK = 60, hK = 40;
  const sepX = (wI + wK) / 2; // 55
  const sepY = (hI + hK) / 2; // 37.5

  // In relative coords (dx, dy) = (x_i - x_k, y_i - y_k)
  // Canvas bounds on macro i: wI/2 <= x_i <= canvasSize - wI/2, hI/2 <= y_i <= canvasSize - hI/2
  // So: (wI/2 - x_k) <= dx <= (canvasSize - wI/2 - x_k), same for dy
  const dxMin = wI / 2 - kx, dxMax = canvasSize - wI / 2 - kx;
  const dyMin = hI / 2 - ky, dyMax = canvasSize - hI / 2 - ky;

  // SVG mapping: dx range [dxMin, dxMax] -> [0, 400], dy range [dyMin, dyMax] -> [400, 0] (inverted y)
  const svgW = 400, svgH = 400;
  const toSvgX = (dx) => (dx - dxMin) / (dxMax - dxMin) * svgW;
  const toSvgY = (dy) => svgH - (dy - dyMin) / (dyMax - dyMin) * svgH;

  // Build each polyhedron as clipped polygon in (dx, dy) space
  const buildPoly = (rel) => {
    let p = [[dxMin, dyMin], [dxMax, dyMin], [dxMax, dyMax], [dxMin, dyMax]];
    switch (rel) {
      case "L": p = clipPolygon(p, ([dx]) => dx <= -sepX); break;
      case "R": p = clipPolygon(p, ([dx]) => dx >= sepX); break;
      case "A": p = clipPolygon(p, ([, dy]) => dy >= sepY); break;
      case "B": p = clipPolygon(p, ([, dy]) => dy <= -sepY); break;
    }
    return p.map(([dx, dy]) => [toSvgX(dx), toSvgY(dy)]);
  };

  const rels = ["L", "R", "A", "B"];
  const polyhedra = {};
  rels.forEach(r => { polyhedra[r] = buildPoly(r); });

  const originSx = toSvgX(0), originSy = toSvgY(0);

  return (
    <div>
      <p style={{ color: COLORS.textMuted, fontSize: 13, marginBottom: 12, lineHeight: 1.6 }}>
        <strong style={{ color: COLORS.text }}>2D, two macros -- all 4 polyhedra.</strong> Fix macro k at ({kx}, {ky}).
        In relative coordinates (dx, dy) = (x_i - x_k, y_i - y_k), each relation carves out a bounded convex polygon.
        Canvas bounds clip the half-planes into finite polyhedra. The exclusion zone in the center is where overlap would occur.
        Click a relation to highlight it, or "All" to see all four.
      </p>

      <div style={{ display: "flex", gap: 6, marginBottom: 8, flexWrap: "wrap", alignItems: "center" }}>
        <button onClick={() => setActiveRel(null)} style={{ padding: "4px 10px", borderRadius: 5, border: `1px solid ${activeRel === null ? "#8b5cf6" : COLORS.grid}`, background: activeRel === null ? "#8b5cf633" : "transparent", color: COLORS.text, fontSize: 11, cursor: "pointer" }}>All</button>
        {rels.map(r => (
          <button key={r} onClick={() => setActiveRel(r)} style={{ padding: "4px 10px", borderRadius: 5, border: `1px solid ${activeRel === r ? REL_COLORS[r].border : COLORS.grid}`, background: activeRel === r ? REL_COLORS[r].fill : "transparent", color: COLORS.text, fontSize: 11, cursor: "pointer" }}>{REL_LABELS[r]}</button>
        ))}
        <span style={{ color: COLORS.textMuted, fontSize: 11, marginLeft: 8 }}>k position:</span>
        <input type="range" min={60} max={340} value={kx} onChange={e => setKx(+e.target.value)} style={{ width: 80 }} />
        <span style={{ color: COLORS.textMuted, fontSize: 10 }}>x={kx}</span>
        <input type="range" min={60} max={340} value={ky} onChange={e => setKy(+e.target.value)} style={{ width: 80 }} />
        <span style={{ color: COLORS.textMuted, fontSize: 10 }}>y={ky}</span>
      </div>

      <svg width="100%" viewBox={`0 0 ${svgW} ${svgH}`} style={{ background: COLORS.bg, borderRadius: 8 }}>
        <defs>
          <pattern id="grid2d" width={svgW / 10} height={svgH / 10} patternUnits="userSpaceOnUse">
            <path d={`M ${svgW / 10} 0 L 0 0 0 ${svgH / 10}`} fill="none" stroke={COLORS.grid} strokeWidth={0.5} />
          </pattern>
        </defs>
        <rect width={svgW} height={svgH} fill="url(#grid2d)" />

        {/* Exclusion zone */}
        {(() => {
          const ex1 = toSvgX(-sepX), ex2 = toSvgX(sepX);
          const ey1 = toSvgY(sepY), ey2 = toSvgY(-sepY);
          return <rect x={Math.min(ex1, ex2)} y={Math.min(ey1, ey2)} width={Math.abs(ex2 - ex1)} height={Math.abs(ey2 - ey1)} fill="rgba(239,68,68,0.08)" stroke="rgba(239,68,68,0.25)" strokeWidth={1} strokeDasharray="4,4" />;
        })()}

        {/* Draw polyhedra */}
        {rels.map(r => {
          const pts = polyhedra[r];
          if (pts.length < 3) return null;
          const show = activeRel === null || activeRel === r;
          const dim = activeRel !== null && activeRel !== r;
          return (
            <g key={r}>
              <polygon
                points={pts.map(p => p.join(",")).join(" ")}
                fill={show ? REL_COLORS[r].fill : "transparent"}
                stroke={REL_COLORS[r].border}
                strokeWidth={show ? 2 : 0.5}
                opacity={dim ? 0.15 : 1}
              />
              {show && pts.length > 2 && (() => {
                const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
                const cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
                return <text x={cx} y={cy + 4} textAnchor="middle" fill={REL_COLORS[r].border} fontSize={13} fontWeight="bold">P_{r}</text>;
              })()}
            </g>
          );
        })}

        {/* Origin marker (where k is) */}
        <circle cx={originSx} cy={originSy} r={4} fill={COLORS.macroB} stroke="white" strokeWidth={1.5} />
        <text x={originSx + 8} y={originSy - 6} fill={COLORS.macroB} fontSize={10} fontWeight="bold">k at origin</text>

        {/* Axes through origin */}
        <line x1={0} y1={originSy} x2={svgW} y2={originSy} stroke={COLORS.axis} strokeWidth={0.5} strokeDasharray="3,3" />
        <line x1={originSx} y1={0} x2={originSx} y2={svgH} stroke={COLORS.axis} strokeWidth={0.5} strokeDasharray="3,3" />

        <text x={svgW / 2} y={svgH - 6} textAnchor="middle" fill={COLORS.textMuted} fontSize={11}>dx = x_i - x_k</text>
        <text x={8} y={svgH / 2} fill={COLORS.textMuted} fontSize={11} transform={`rotate(-90, 8, ${svgH / 2})`}>dy = y_i - y_k</text>

        <text x={svgW / 2} y={16} textAnchor="middle" fill={COLORS.text} fontSize={13} fontWeight="bold">
          4 Polyhedra in Relative Position Space
        </text>

        {/* Exclusion zone label */}
        <text x={originSx} y={originSy + 16} textAnchor="middle" fill="rgba(239,68,68,0.5)" fontSize={9}>exclusion zone</text>
      </svg>

      <p style={{ color: COLORS.textMuted, fontSize: 12, marginTop: 8, lineHeight: 1.5 }}>
        Move k's position with the sliders. Notice how the polyhedra reshape: when k is near a corner, the polyhedron
        on that side shrinks (less room for i). When k is centered, all four polyhedra are roughly symmetric.
        Each polyhedron has <strong style={{ color: COLORS.text }}>3-5 faces</strong>: one from the separation constraint
        (the side facing the exclusion zone) and the rest from canvas bounds.
      </p>
    </div>
  );
}

// ============== Tab 3: 2D Three Macros -- feasible region for C given A,B fixed ==============
function TwoDThreeMacros() {
  const [assignment, setAssignment] = useState({ ab: "L", ac: "R", bc: "B" });
  const [dragging, setDragging] = useState(null);
  const [posA, setPosA] = useState([100, 180]);
  const [posB, setPosB] = useState([280, 180]);

  const canvasSize = 400;
  const dims = { a: [50, 35], b: [60, 40], c: [45, 45] };

  // Separation distances
  const sepAC_x = (dims.a[0] + dims.c[0]) / 2; // 47.5
  const sepAC_y = (dims.a[1] + dims.c[1]) / 2; // 40
  const sepBC_x = (dims.b[0] + dims.c[0]) / 2; // 52.5
  const sepBC_y = (dims.b[1] + dims.c[1]) / 2; // 42.5

  // Canvas bounds for C center
  const cxMin = dims.c[0] / 2, cxMax = canvasSize - dims.c[0] / 2;
  const cyMin = dims.c[1] / 2, cyMax = canvasSize - dims.c[1] / 2;

  // Build feasible region for C as intersection of canvas bounds + AC constraint + BC constraint
  const buildFeasibleC = () => {
    // Start with canvas bounds rectangle for C
    let p = [[cxMin, cyMin], [cxMax, cyMin], [cxMax, cyMax], [cxMin, cyMax]];

    // AC constraint
    const [ax, ay] = posA;
    switch (assignment.ac) {
      case "L": p = clipPolygon(p, ([cx]) => ax + dims.a[0] / 2 <= cx - dims.c[0] / 2); break; // A left of C: cx >= ax + sepAC_x
      case "R": p = clipPolygon(p, ([cx]) => cx + dims.c[0] / 2 <= ax - dims.a[0] / 2); break; // A right of C: cx <= ax - sepAC_x
      case "A": p = clipPolygon(p, ([, cy]) => ay - dims.a[1] / 2 >= cy + dims.c[1] / 2); break; // A above C: cy <= ay - sepAC_y
      case "B": p = clipPolygon(p, ([, cy]) => cy - dims.c[1] / 2 >= ay + dims.a[1] / 2); break; // A below C: cy >= ay + sepAC_y
    }

    // BC constraint
    const [bx, by] = posB;
    switch (assignment.bc) {
      case "L": p = clipPolygon(p, ([cx]) => bx + dims.b[0] / 2 <= cx - dims.c[0] / 2); break;
      case "R": p = clipPolygon(p, ([cx]) => cx + dims.c[0] / 2 <= bx - dims.b[0] / 2); break;
      case "A": p = clipPolygon(p, ([, cy]) => by - dims.b[1] / 2 >= cy + dims.c[1] / 2); break;
      case "B": p = clipPolygon(p, ([, cy]) => cy - dims.c[1] / 2 >= by + dims.b[1] / 2); break;
    }

    return p;
  };

  // Check if A-B assignment is satisfied
  const checkAB = () => {
    const [ax, ay] = posA, [bx, by] = posB;
    switch (assignment.ab) {
      case "L": return ax + dims.a[0] / 2 <= bx - dims.b[0] / 2;
      case "R": return bx + dims.b[0] / 2 <= ax - dims.a[0] / 2;
      case "A": return ay - dims.a[1] / 2 >= by + dims.b[1] / 2;
      case "B": return by - dims.b[1] / 2 >= ay + dims.a[1] / 2;
    }
  };

  const abOk = checkAB();
  const feasibleC = buildFeasibleC();
  const feasible = abOk && feasibleC.length >= 3;

  // SVG mapping: chip coords [0, canvasSize] -> SVG [0, 400], y inverted
  const S = 400;
  const toSvg = ([x, y]) => [x, S - y];

  const handleMouseDown = (which) => (e) => { e.preventDefault(); setDragging(which); };
  const handleMouseMove = (e) => {
    if (!dragging) return;
    const svg = e.currentTarget;
    const rect = svg.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * S;
    const y = S - ((e.clientY - rect.top) / rect.height) * S;
    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
    if (dragging === "A") setPosA([clamp(x, dims.a[0] / 2, canvasSize - dims.a[0] / 2), clamp(y, dims.a[1] / 2, canvasSize - dims.a[1] / 2)]);
    if (dragging === "B") setPosB([clamp(x, dims.b[0] / 2, canvasSize - dims.b[0] / 2), clamp(y, dims.b[1] / 2, canvasSize - dims.b[1] / 2)]);
  };

  const pairColors = { ab: "#3b82f6", ac: "#f59e0b", bc: "#10b981" };

  return (
    <div>
      <p style={{ color: COLORS.textMuted, fontSize: 13, marginBottom: 12, lineHeight: 1.6 }}>
        <strong style={{ color: COLORS.text }}>2D, three macros -- the polyhedron as a feasible region.</strong> Fix A and B
        (drag them). Choose L/R/A/B for all three pairs. The <span style={{ color: "#8b5cf6" }}>purple shaded region</span> shows
        where C can legally be placed -- this is a 2D slice of the 6D polyhedron for this assignment.
        The intersection of two pair constraints plus canvas bounds creates a convex polygon.
      </p>

      <div style={{ marginBottom: 10 }}>
        {[["ab", "A-B"], ["ac", "A-C"], ["bc", "B-C"]].map(([pair, label]) => (
          <div key={pair} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
            <span style={{ color: pairColors[pair], fontSize: 12, fontWeight: "bold", width: 36 }}>{label}:</span>
            {["L", "R", "A", "B"].map(rel => (
              <button key={rel} onClick={() => setAssignment(prev => ({ ...prev, [pair]: rel }))}
                style={{ padding: "2px 8px", borderRadius: 4, fontSize: 10, cursor: "pointer",
                  border: `1px solid ${assignment[pair] === rel ? pairColors[pair] : COLORS.grid}`,
                  background: assignment[pair] === rel ? pairColors[pair] + "33" : "transparent",
                  color: assignment[pair] === rel ? "white" : COLORS.textMuted }}>
                {rel}
              </button>
            ))}
          </div>
        ))}
      </div>

      <svg width="100%" viewBox={`0 0 ${S} ${S}`}
        style={{ background: COLORS.bg, borderRadius: 8, cursor: dragging ? "grabbing" : "default" }}
        onMouseMove={handleMouseMove} onMouseUp={() => setDragging(null)} onMouseLeave={() => setDragging(null)}>
        <defs>
          <pattern id="grid3" width={S / 10} height={S / 10} patternUnits="userSpaceOnUse">
            <path d={`M ${S / 10} 0 L 0 0 0 ${S / 10}`} fill="none" stroke={COLORS.grid} strokeWidth={0.5} />
          </pattern>
        </defs>
        <rect width={S} height={S} fill="url(#grid3)" />

        {/* Feasible region for C */}
        {feasibleC.length >= 3 && (
          <polygon
            points={feasibleC.map(pt => toSvg(pt).join(",")).join(" ")}
            fill="rgba(139, 92, 246, 0.25)"
            stroke="rgba(139, 92, 246, 0.8)"
            strokeWidth={2}
          />
        )}

        {/* Macro A */}
        {(() => {
          const [sx, sy] = toSvg(posA);
          return (
            <g style={{ cursor: "grab" }} onMouseDown={handleMouseDown("A")}>
              <rect x={sx - dims.a[0] / 2} y={sy - dims.a[1] / 2} width={dims.a[0]} height={dims.a[1]} rx={3}
                fill={abOk ? "rgba(59,130,246,0.35)" : "rgba(239,68,68,0.35)"} stroke={abOk ? COLORS.macroA : "#ef4444"} strokeWidth={2} />
              <text x={sx} y={sy + 4} textAnchor="middle" fill="white" fontSize={12} fontWeight="bold" pointerEvents="none">A</text>
            </g>
          );
        })()}

        {/* Macro B */}
        {(() => {
          const [sx, sy] = toSvg(posB);
          return (
            <g style={{ cursor: "grab" }} onMouseDown={handleMouseDown("B")}>
              <rect x={sx - dims.b[0] / 2} y={sy - dims.b[1] / 2} width={dims.b[0]} height={dims.b[1]} rx={3}
                fill={abOk ? "rgba(245,158,11,0.35)" : "rgba(239,68,68,0.35)"} stroke={abOk ? COLORS.macroB : "#ef4444"} strokeWidth={2} />
              <text x={sx} y={sy + 4} textAnchor="middle" fill="white" fontSize={12} fontWeight="bold" pointerEvents="none">B</text>
            </g>
          );
        })()}

        {/* Status */}
        <text x={S / 2} y={16} textAnchor="middle" fill={COLORS.text} fontSize={13} fontWeight="bold">
          Feasible Region for C (2D slice of 6D polyhedron)
        </text>
        {!abOk && <text x={S / 2} y={36} textAnchor="middle" fill="#ef4444" fontSize={11}>A-B constraint violated -- move them apart</text>}
        {abOk && feasibleC.length < 3 && <text x={S / 2} y={36} textAnchor="middle" fill="#ef4444" fontSize={11}>Empty polyhedron -- this assignment is infeasible for these positions</text>}
        {feasible && <text x={S / 2} y={36} textAnchor="middle" fill="#22c55e" fontSize={11}>
          Polyhedron has {feasibleC.length} vertices -- C can move freely inside
        </text>}
      </svg>

      <div style={{ background: "#1a1a2e", borderRadius: 8, padding: 10, fontSize: 11, color: COLORS.textMuted, lineHeight: 1.6, marginTop: 10 }}>
        <strong style={{ color: COLORS.text }}>What you are seeing:</strong> The purple polygon is the intersection of
        (1) canvas bounds for C, (2) the A-C separation half-plane, and (3) the B-C separation half-plane.
        Try moving A and B closer together or to corners -- watch the feasible region shrink, change shape, or
        vanish entirely (infeasible assignment). Each vertex of the polygon is a point where two or more
        constraints are simultaneously tight.
      </div>
    </div>
  );
}

// ============== Tab 4: Polyhedra Union ==============
function PolyhedraUnion() {
  const [highlightIdx, setHighlightIdx] = useState(0);

  const orderings = [
    { label: "A-B-C", constraints: "x_A < x_B < x_C", color: "#3b82f6" },
    { label: "A-C-B", constraints: "x_A < x_C < x_B", color: "#8b5cf6" },
    { label: "B-A-C", constraints: "x_B < x_A < x_C", color: "#ec4899" },
    { label: "B-C-A", constraints: "x_B < x_C < x_A", color: "#f59e0b" },
    { label: "C-A-B", constraints: "x_C < x_A < x_B", color: "#10b981" },
    { label: "C-B-A", constraints: "x_C < x_B < x_A", color: "#ef4444" },
  ];

  return (
    <div>
      <p style={{ color: COLORS.textMuted, fontSize: 13, marginBottom: 12, lineHeight: 1.6 }}>
        <strong style={{ color: COLORS.text }}>Union of polyhedra.</strong> For 3 macros in 1D, there are 3! = 6 orderings.
        Each ordering defines a cone (polyhedron) in R^3 position space. The feasible region is the union
        of these 6 non-overlapping cones. The gaps between them are infeasible overlap regions. Moving between
        cones requires crossing through infeasible space.
      </p>

      <svg width="100%" viewBox="0 0 500 320" style={{ background: COLORS.bg, borderRadius: 8, marginBottom: 12 }}>
        <text x={250} y={24} textAnchor="middle" fill={COLORS.text} fontSize={13} fontWeight="bold">
          Schematic: 6 Polyhedra (cones in R^3) with Infeasible Gaps
        </text>
        {orderings.map((ord, i) => {
          const angle = (i / 6) * Math.PI * 2 - Math.PI / 2;
          const r1 = 50, r2 = 120, aSpan = Math.PI / 6 * 0.7, cx = 250, cy = 175;
          const x1 = cx + r1 * Math.cos(angle - aSpan), y1 = cy + r1 * Math.sin(angle - aSpan);
          const x2 = cx + r2 * Math.cos(angle - aSpan), y2 = cy + r2 * Math.sin(angle - aSpan);
          const x3 = cx + r2 * Math.cos(angle + aSpan), y3 = cy + r2 * Math.sin(angle + aSpan);
          const x4 = cx + r1 * Math.cos(angle + aSpan), y4 = cy + r1 * Math.sin(angle + aSpan);
          const isActive = highlightIdx === i;
          const lx = cx + (r2 + 25) * Math.cos(angle), ly = cy + (r2 + 25) * Math.sin(angle);
          return (
            <g key={i} style={{ cursor: "pointer" }} onClick={() => setHighlightIdx(i)}>
              <polygon points={`${x1},${y1} ${x2},${y2} ${x3},${y3} ${x4},${y4}`}
                fill={isActive ? ord.color + "55" : ord.color + "22"} stroke={ord.color} strokeWidth={isActive ? 2.5 : 1} />
              <text x={lx} y={ly} textAnchor="middle" dominantBaseline="middle"
                fill={isActive ? "white" : ord.color} fontSize={isActive ? 12 : 10} fontWeight={isActive ? "bold" : "normal"}>
                {ord.label}
              </text>
            </g>
          );
        })}
        <text x={250} y={175} textAnchor="middle" fill="rgba(239,68,68,0.4)" fontSize={10} fontStyle="italic">infeasible gaps</text>
        <rect x={20} y={265} width={460} height={45} rx={6} fill="#1a1a2e" />
        <text x={250} y={282} textAnchor="middle" fill={orderings[highlightIdx].color} fontSize={12} fontWeight="bold">
          Polyhedron "{orderings[highlightIdx].label}": {orderings[highlightIdx].constraints}
        </text>
        <text x={250} y={300} textAnchor="middle" fill={COLORS.textMuted} fontSize={11}>
          Within this cone, all 3 macros can slide continuously without overlapping.
        </text>
      </svg>

      <div style={{ background: "#1a1a2e", borderRadius: 8, padding: 12, fontSize: 12, color: COLORS.textMuted, lineHeight: 1.6 }}>
        <strong style={{ color: COLORS.text }}>Scaling:</strong> With n macros in 1D: n! orderings. In 2D with 4 relations per pair: up to 4^(n*(n-1)/2) assignments.
        For n=300: 4^44,850 polyhedra. Most are infeasible, but the feasible count is still astronomically large.
      </div>
    </div>
  );
}

// ============== Tab 5: Faces and Edges ==============
function FacesAndEdges() {
  return (
    <div>
      <p style={{ color: COLORS.textMuted, fontSize: 13, marginBottom: 12, lineHeight: 1.6 }}>
        <strong style={{ color: COLORS.text }}>Anatomy of a polyhedron.</strong> Each face corresponds to one constraint being
        tight (active at equality). Understanding faces is key to navigation between polyhedra.
      </p>

      <svg width="100%" viewBox="0 0 500 420" style={{ background: COLORS.bg, borderRadius: 8 }}>
        <text x={250} y={24} textAnchor="middle" fill={COLORS.text} fontSize={13} fontWeight="bold">
          Faces of a Placement Polyhedron
        </text>
        <polygon points="150,80 320,60 420,150 400,300 230,350 100,280"
          fill="rgba(139,92,246,0.1)" stroke="rgba(139,92,246,0.5)" strokeWidth={2} />

        {/* Face labels */}
        <line x1={150} y1={80} x2={320} y2={60} stroke="#60a5fa" strokeWidth={3} />
        <line x1={235} y1={70} x2={235} y2={35} stroke="#60a5fa" strokeWidth={1} strokeDasharray="2,2" />
        <text x={235} y={28} textAnchor="middle" fill="#60a5fa" fontSize={10} fontWeight="bold">Canvas top bound</text>
        <text x={235} y={44} textAnchor="middle" fill="#60a5fa" fontSize={9} opacity={0.7}>y_i + h_i/2 = H</text>

        <line x1={320} y1={60} x2={420} y2={150} stroke="#f59e0b" strokeWidth={3} />
        <line x1={370} y1={105} x2={440} y2={80} stroke="#f59e0b" strokeWidth={1} strokeDasharray="2,2" />
        <text x={465} y={73} textAnchor="middle" fill="#f59e0b" fontSize={10} fontWeight="bold">Pair (i,k) touching</text>
        <text x={465} y={89} textAnchor="middle" fill="#f59e0b" fontSize={9} opacity={0.7}>x_k - x_i = sep_x</text>

        <line x1={420} y1={150} x2={400} y2={300} stroke="#10b981" strokeWidth={3} />
        <line x1={410} y1={225} x2={460} y2={225} stroke="#10b981" strokeWidth={1} strokeDasharray="2,2" />
        <text x={475} y={218} textAnchor="middle" fill="#10b981" fontSize={10} fontWeight="bold">Pair (j,m)</text>
        <text x={475} y={234} textAnchor="middle" fill="#10b981" fontSize={9} opacity={0.7}>y_m - y_j = sep_y</text>

        <line x1={400} y1={300} x2={230} y2={350} stroke="#60a5fa" strokeWidth={3} />
        <line x1={315} y1={325} x2={315} y2={370} stroke="#60a5fa" strokeWidth={1} strokeDasharray="2,2" />
        <text x={315} y={383} textAnchor="middle" fill="#60a5fa" fontSize={10} fontWeight="bold">Canvas bottom bound</text>
        <text x={315} y={397} textAnchor="middle" fill="#60a5fa" fontSize={9} opacity={0.7}>y_j - h_j/2 = 0</text>

        <line x1={230} y1={350} x2={100} y2={280} stroke="#ec4899" strokeWidth={3} />
        <line x1={165} y1={315} x2={60} y2={340} stroke="#ec4899" strokeWidth={1} strokeDasharray="2,2" />
        <text x={55} y={355} textAnchor="start" fill="#ec4899" fontSize={10} fontWeight="bold">Pair (i,j) touching</text>
        <text x={55} y={369} textAnchor="start" fill="#ec4899" fontSize={9} opacity={0.7}>x_i - x_j = sep_x</text>

        <line x1={100} y1={280} x2={150} y2={80} stroke="#a78bfa" strokeWidth={3} />
        <line x1={125} y1={180} x2={50} y2={180} stroke="#a78bfa" strokeWidth={1} strokeDasharray="2,2" />
        <text x={45} y={173} textAnchor="end" fill="#a78bfa" fontSize={10} fontWeight="bold">Canvas left</text>
        <text x={45} y={189} textAnchor="end" fill="#a78bfa" fontSize={9} opacity={0.7}>x_i - w_i/2 = 0</text>

        <text x={260} y={195} textAnchor="middle" fill="rgba(139,92,246,0.6)" fontSize={12} fontWeight="bold">Interior:</text>
        <text x={260} y={211} textAnchor="middle" fill="rgba(139,92,246,0.5)" fontSize={10}>all constraints slack</text>
        <text x={260} y={225} textAnchor="middle" fill="rgba(139,92,246,0.5)" fontSize={10}>(macros have room to move)</text>

        <circle cx={420} cy={150} r={5} fill="#ef4444" />
        <text x={420} y={165} textAnchor="middle" fill="#ef4444" fontSize={9}>vertex: multiple faces meet</text>
      </svg>

      <div style={{ background: "#1a1a2e", borderRadius: 8, padding: 12, fontSize: 12, color: COLORS.textMuted, lineHeight: 1.7, marginTop: 12 }}>
        <strong style={{ color: COLORS.text }}>Two types of faces:</strong>
        <div style={{ marginTop: 6 }}>
          <span style={{ color: "#60a5fa" }}>Canvas bounds</span> -- a macro is flush against the canvas edge.
        </div>
        <div style={{ marginTop: 4 }}>
          <span style={{ color: "#f59e0b" }}>Pair separations</span> -- two macros are touching (zero gap).
          This face is shared with an adjacent polyhedron. Crossing it = "flipping" the pair relation.
        </div>
        <div style={{ marginTop: 8 }}>
          <strong style={{ color: COLORS.text }}>LP duals</strong> at an optimal vertex tell you the cost of each tight constraint.
          A large dual on a pair-separation face means that pair is the best candidate to flip.
        </div>
      </div>
    </div>
  );
}

// ============== Main ==============
const TABS = [
  { id: "1d2", label: "1D, 2 Macros", component: OneDTwoMacros },
  { id: "2d2", label: "2D, 2 Macros", component: TwoDTwoMacros },
  { id: "2d3", label: "2D, 3 Macros", component: TwoDThreeMacros },
  { id: "union", label: "Union", component: PolyhedraUnion },
  { id: "faces", label: "Faces & Edges", component: FacesAndEdges },
];

export default function PolyhedraViz() {
  const [activeTab, setActiveTab] = useState("2d2");
  const ActiveComponent = TABS.find(t => t.id === activeTab).component;

  return (
    <div style={{ maxWidth: 600, margin: "0 auto", padding: 16, fontFamily: "system-ui, -apple-system, sans-serif", color: COLORS.text }}>
      <h2 style={{ fontSize: 18, fontWeight: "bold", marginBottom: 4, color: COLORS.text }}>
        Polyhedra in Macro Placement
      </h2>
      <p style={{ color: COLORS.textMuted, fontSize: 12, marginBottom: 16 }}>
        How disjunctive non-overlap constraints create convex polyhedra whose union is the feasible region
      </p>
      <div style={{ display: "flex", gap: 4, marginBottom: 16, flexWrap: "wrap" }}>
        {TABS.map(tab => (
          <button key={tab.id} onClick={() => setActiveTab(tab.id)}
            style={{ padding: "6px 12px", borderRadius: 6, border: "none", fontSize: 12, cursor: "pointer",
              background: activeTab === tab.id ? "#8b5cf6" : "#1e293b",
              color: activeTab === tab.id ? "white" : COLORS.textMuted,
              fontWeight: activeTab === tab.id ? "bold" : "normal" }}>
            {tab.label}
          </button>
        ))}
      </div>
      <ActiveComponent />
    </div>
  );
}

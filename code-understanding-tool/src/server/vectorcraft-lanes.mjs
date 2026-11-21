import assert from "node:assert/strict";

export function laneLayout(visual) {
  const nodes = new Map(visual.nodes.map((node) => [node.id, node]));
  assert(nodes.size === visual.nodes.length && nodes.size > 0 && nodes.size <= 16, "Expected 1-16 distinct nodes.");
  const outgoing = new Map([...nodes.keys()].map((id) => [id, []]));
  for (const edge of visual.edges) {
    assert(nodes.has(edge.from) && nodes.has(edge.to), "Unknown edge endpoint.");
    assert(!outgoing.get(edge.from).some((other) => other.to === edge.to), "Parallel edges are unsupported.");
    outgoing.get(edge.from).push(edge);
  }
  assert([...outgoing.values()].every((edges) => edges.length <= 2), "Only binary decisions are supported.");
  const targets = new Set(visual.edges.map((edge) => edge.to));
  const roots = [...nodes.keys()].filter((id) => !targets.has(id));
  assert.equal(roots.length, 1, "Expected a single entry.");
  const reachable = new Map();
  function distances(id, active = new Set()) {
    assert(!active.has(id), "Cycles are unsupported.");
    if (reachable.has(id)) return reachable.get(id);
    const found = new Map([[id, 0]]);
    active.add(id);
    for (const edge of outgoing.get(id)) for (const [next, distance] of distances(edge.to, active)) {
      found.set(next, Math.min(found.get(next) ?? Infinity, distance + 1));
    }
    active.delete(id);
    reachable.set(id, found);
    return found;
  }
  assert.equal(distances(roots[0]).size, nodes.size, "Graph must be connected and acyclic.");
  const visited = new Set();

  // Parse nested split/join blocks; branches cannot share a node before their join.
  function sequence(start, stop) {
    const rows = [];
    for (let id = start; id !== stop;) {
      assert(id !== undefined && !visited.has(id), "Branches must meet at their declared join.");
      visited.add(id);
      const edges = outgoing.get(id);
      if (edges.length < 2) {
        rows.push({ id, width: 1, height: 1 });
        id = edges[0]?.to;
        continue;
      }
      const left = distances(edges[0].to), right = distances(edges[1].to);
      const joined = [...left.keys()].filter((candidate) => right.has(candidate))
        .sort((a, b) => left.get(a) + right.get(a) - left.get(b) - right.get(b))[0];
      assert(joined !== undefined && edges.every((edge) => edge.to !== joined), "Expected two nonempty branches with a shared join.");
      const branches = edges.map((edge) => sequence(edge.to, joined));
      rows.push({ id, branches, width: branches[0].width + branches[1].width,
        height: 1 + Math.max(...branches.map((branch) => branch.height)) });
      id = joined;
    }
    return { rows, width: Math.max(1, ...rows.map((row) => row.width)), height: rows.reduce((sum, row) => sum + row.height, 0) };
  }
  const tree = sequence(roots[0], undefined);
  assert.equal(visited.size, nodes.size);
  const positions = new Map();
  function place(seq, left, y, lane) {
    const center = left + seq.width / 2;
    for (const row of seq.rows) {
      positions.set(row.id, { x: center, y, lane, decision: Boolean(row.branches) });
      if (row.branches) {
        let start = center - row.width / 2;
        row.branches.forEach((branch, index) => {
          place(branch, start, y + 1, lane ?? index);
          start += branch.width;
        });
      }
      y += row.height;
    }
  }
  place(tree, 0, 0, undefined);
  return { positions, width: tree.width, height: tree.height, edges: visual.edges };
}

function wrap(value, limit) {
  const lines = [""];
  for (const word of value.trim().split(/\s+/)) {
    assert(Array.from(word).length <= limit, "Label has an unsupported long word.");
    if (Array.from((lines.at(-1) + " " + word).trim()).length > limit) lines.push(word);
    else lines[lines.length - 1] = (lines.at(-1) + " " + word).trim();
  }
  return lines;
}

export function laneCommands(visual, sourceLabel, lineLabel = "saved chapter patch") {
  const layout = laneLayout(visual);
  const commands = [];
  const width = Math.max(960, layout.width * 360 + 96);
  const rowGap = 132;
  const height = 311 + (layout.height - 1) * rowGap;
  const step = (width - 96) / layout.width;
  const position = (id) => {
    const p = layout.positions.get(id);
    return { ...p, x: 48 + p.x * step, y: 202 + p.y * rowGap };
  };
  const add = (command, params) => commands.push({ command, params });
  const fill = (color) => add("paint.setFill", { color });
  const stroke = (color, arrow = false) => {
    add("paint.setStroke", { color });
    add("stroke.set", { weight: 2, startArrow: null, endArrow: arrow ? "Triangle" : null, arrowAlign: "tip" });
  };
  const path = (points, color, { arrow = false, closed = false } = {}) => {
    add("path.create", { anchors: points.map(([x, y]) => ({ x, y })), closed });
    add("paint.setFill", { none: true });
    stroke(color, arrow);
  };
  const box = (x, y, width, height, color, border) => {
    add("shape.rectangle", { x, y, width, height });
    fill(color);
    if (border) stroke(border);
    else add("paint.setStroke", { none: true });
  };
  const text = (value, x, y, size, center = false, color = "#1b3934") => {
    add("text.create", { text: value, x, y, size, font: "Source Sans 3", color, justify: center === "right" ? "right" : center ? "center" : "left" });
    add("paint.setStroke", { none: true });
  };
  const color = (lane) => lane === 1 ? "#476e8a" : "#367364";
  add("file.new", { width, height, name: "Branch layout prototype", backgroundContents: "white", units: "Points" });
  box(0, 0, width, height, "#f7faf8");
  text(sourceLabel, 48, 44, 16);
  text(visual.title, 48, 97, 36);
  text("Source-linked branch view  /  Not a runtime trace", 48, 130, 20);
  for (const edge of layout.edges) {
    const from = position(edge.from), to = position(edge.to);
    const ink = color(to.lane ?? from.lane);
    const endY = to.y - (to.decision ? 53 : 40);
    const startY = from.y + (from.decision ? 53 : 40);
    if (from.decision) {
      const direction = Math.sign(to.x - from.x);
      const startX = from.x + direction * 146;
      path([[startX, from.y], [to.x, from.y], [to.x, endY]], ink, { arrow: true });
      const labels = wrap(edge.label, 28);
      assert(labels.length <= 2, "Branch label is too long.");
      labels.forEach((line, i) => text(line, to.x + direction * 13, endY - 18 - (labels.length - 1 - i) * 18, 16, direction < 0 ? "right" : false, ink));
    } else {
      const turnY = endY - 23;
      path(from.x === to.x ? [[from.x, startY], [to.x, endY]] : [[from.x, startY], [from.x, turnY], [to.x, turnY], [to.x, endY]], ink, { arrow: true });
    }
  }
  for (const node of visual.nodes) {
    const p = position(node.id), ink = color(p.lane);
    const lines = wrap(node.label.replaceAll("`", ""), p.decision ? 24 : 28);
    assert(lines.length <= 2, "Node label is too long.");
    if (p.decision) {
      path([[p.x, p.y-53], [p.x+146,p.y], [p.x,p.y+53], [p.x-146,p.y]], ink, { closed: true });
      fill(p.lane === 1 ? "#e4eef5" : "#e0efe7");
    } else box(p.x-146, p.y-40, 292, 80, "#ffffff", ink);
    lines.forEach((line, i) => text(line, p.x, p.y - (lines.length === 2 ? 12 : 4) + i * 19, 18, true));
    const refs = node.source_ranges.map((r) => r.start_line === r.end_line ? r.start_line : r.start_line + "-" + r.end_line).join(", ");
    assert(refs.length <= 27, "Source citation is too long.");
    text("Lines " + (refs || "not supplied"), p.x, p.y + (p.decision ? 24 : 28), 11, true, ink);
  }
  text("Selected relationships only. Line numbers refer to the " + lineLabel + ".", 48, height - 25, 16);
  return { commands, layout, width, height, mode: "lanes" };
}

export function diagramPlan(visual, sourceLabel, lineLabel = "saved chapter patch") {
  const ids = new Map(visual.nodes.map((node, index) => [node.id, index + 1]));
  assert(ids.size > 0 && ids.size === visual.nodes.length && ids.size <= 30, "Expected 1-30 distinct nodes.");
  assert(visual.edges.length <= 60, "Too many connections.");
  for (const edge of visual.edges) assert(ids.has(edge.from) && ids.has(edge.to), "Unknown edge endpoint.");
  try {
    const plan = laneCommands(visual, sourceLabel, lineLabel);
    if (plan.width <= 1800) return plan;
  } catch (error) {
    if (error.code !== "ERR_ASSERTION") throw error;
  }

  // Use an exact connection index when drawing lanes would misrepresent the graph.
  const width = 1100;
  const commands = [];
  const add = (command, params) => commands.push({ command, params });
  const text = (value, x, y, size = 17, color = "#1b3934") => {
    add("text.create", { text: value, x, y, size, font: "Source Sans 3", color });
    add("paint.setStroke", { none: true });
  };
  const wrapRow = (value, limit) => wrap(value.replace(/\S+/gu, (word) => {
    const letters = Array.from(word);
    return Array.from({ length: Math.ceil(letters.length / limit) }, (_, index) => letters.slice(index * limit, (index + 1) * limit).join("")).join(" ");
  }), limit);
  const band = (x, y, width, height) => {
    add("shape.rectangle", { x, y, width, height });
    add("paint.setFill", { color: "#e8f1ed" });
    add("paint.setStroke", { none: true });
  };
  function column(entries, x) {
    let y = 223;
    for (const [label, detail] of entries) {
      const labels = wrapRow(label, 39);
      const details = wrapRow(detail, 49);
      const rowHeight = 26 + labels.length * 22 + details.length * 19;
      band(x, y, 494, rowHeight);
      labels.forEach((line, i) => text(line, x + 18, y + 25 + i * 22, 19));
      details.forEach((line, i) => text(line, x + 18, y + 27 + labels.length * 22 + i * 19, 15, "#4c695e"));
      y += rowHeight + 8;
    }
    return y;
  }
  text(sourceLabel, 48, 42, 16);
  wrapRow(visual.title, 58).forEach((line, i) => {
    assert(i < 2, "Title is too long.");
    text(line, 48, 86 + i * 34, 31);
  });
  text("Connection index fallback  /  Node numbers are references, not execution order.", 48, 153, 18);
  text("NODES AND SOURCE LINES", 48, 201, 15);
  text("DIRECTED CONNECTIONS", 558, 201, 15);
  const left = column(visual.nodes.map((node) => {
    const refs = node.source_ranges.map(({ start_line, end_line }) => start_line === end_line ? start_line : start_line + "-" + end_line);
    return [ids.get(node.id) + "  " + node.label, "Lines " + (refs.join(", ") || "not supplied")];
  }), 48);
  const right = column(visual.edges.map(({ from, to, label }, i) => ["E" + (i + 1) + "  Node " + ids.get(from) + " -> Node " + ids.get(to), label || "Unlabelled connection"]), 558);
  if (!visual.edges.length) text("No supplied connections.", 558, 247);
  const height = Math.max(left, right) + 62;
  assert(height <= 8000, "Connection index is too tall.");
  text("Selected relationships only. Line numbers refer to the " + lineLabel + ".", 48, height - 25, 16);
  commands.unshift({ command: "file.new", params: { width, height, name: "Connection index", backgroundContents: "white", units: "Points" } });
  return { commands, layout: { positions: ids, edges: visual.edges }, width, height, mode: "connection-index" };
}

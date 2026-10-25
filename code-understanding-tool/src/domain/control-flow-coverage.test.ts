import { describe, expect, it } from "vitest";
import { analyzeControlFlowCoverage, countSourceDecisions, hasCompleteControlFlowCoverage } from "./control-flow-coverage";

const shippingSource = `function shippingCost(order: {
  address?: string;
  total: number;
  expedited: boolean;
  member: boolean;
}): number | string {
  if (!order.address) return "Address required";
  let cost = order.total > 100 ? 0 : 12;
  if (order.expedited) cost += 20;
  if (order.member) cost *= 0.9;
  return cost;
}`;

const completeVisual = {
  kind: "control_flow",
  nodes: [
    { id: "address", label: "Check address", source_ranges: [{ start_line: 7, end_line: 7 }] },
    { id: "error", label: "Return error" },
    { id: "total", label: "Check total", source_ranges: [{ start_line: 8, end_line: 8 }] },
    { id: "free", label: "Use zero base" },
    { id: "standard", label: "Use standard base" },
    { id: "express", label: "Check expedited", source_ranges: [{ start_line: 9, end_line: 9 }] },
    { id: "fee", label: "Add fee" },
    { id: "keep", label: "Keep base" },
    { id: "member", label: "Check membership", source_ranges: [{ start_line: 10, end_line: 10 }] },
    { id: "discount", label: "Apply discount" },
    { id: "full", label: "Keep full cost" },
    { id: "return", label: "Return cost" },
  ],
  edges: [
    { from: "address", to: "error", label: "missing" },
    { from: "address", to: "total", label: "present" },
    { from: "total", to: "free", label: "total > 100" },
    { from: "total", to: "standard", label: "total ≤ 100" },
    { from: "free", to: "express", label: "" },
    { from: "standard", to: "express", label: "" },
    { from: "express", to: "fee", label: "expedited" },
    { from: "express", to: "keep", label: "not expedited" },
    { from: "fee", to: "member", label: "" },
    { from: "keep", to: "member", label: "" },
    { from: "member", to: "discount", label: "member" },
    { from: "member", to: "full", label: "not a member" },
    { from: "discount", to: "return", label: "" },
    { from: "full", to: "return", label: "" },
  ],
};

describe("control-flow diagram coverage", () => {
  it("counts if statements and ternaries without counting optional properties", () => {
    expect(countSourceDecisions(shippingSource)).toBe(4);
  });

  it("counts common Python if and conditional-expression forms", () => {
    const pythonSource = "if not address:\n    return 'missing'\ncost = 0 if total > 100 else 12";
    expect(countSourceDecisions(pythonSource)).toBe(2);
  });

  it("accepts a visual with one distinct branch node per source decision", () => {
    expect(hasCompleteControlFlowCoverage(shippingSource, completeVisual)).toBe(true);
  });

  it("matches property checks against the property rather than generic accessors", () => {
    const source = "if (!order.items.length) throw new Error('empty');";
    const visual = {
      kind: "control_flow",
      nodes: [
        { id: "items", label: "Check items", source_ranges: [{ start_line: 1, end_line: 1 }] },
        { id: "empty", label: "Throw error" },
        { id: "continue", label: "Continue" },
      ],
      edges: [
        { from: "items", to: "empty", label: "no items" },
        { from: "items", to: "continue", label: "has items" },
      ],
    };
    expect(hasCompleteControlFlowCoverage(source, visual)).toBe(true);
  });

  it("accepts the order-processing overview and focused shipping view", () => {
    const source = `type Order = {
      items: string[];
      total: number;
      expedited: boolean;
      member: boolean;
      paymentApproved: boolean;
    };
    function processOrder(order: Order): string {
      if (order.items.length === 0) throw new Error("Order is empty");
      if (!order.paymentApproved) return "Payment failed";
      let shipping = order.total > 100 ? 0 : 12;
      if (order.expedited) shipping += 20;
      if (order.member) shipping *= 0.9;
      return \`Confirmed: $\${shipping.toFixed(2)}\`;
    }`;
    const visuals = [
      {
        kind: "control_flow",
        nodes: [
          { id: "process", label: "Process order" }, { id: "items", label: "Check for items", source_ranges: [{ start_line: 9, end_line: 9 }] },
          { id: "empty", label: "Throw empty-order error" }, { id: "payment", label: "Check payment", source_ranges: [{ start_line: 10, end_line: 10 }] },
          { id: "failure", label: "Return payment failure" }, { id: "shipping", label: "Calculate shipping" },
          { id: "confirm", label: "Return confirmation" },
        ],
        edges: [
          { from: "process", to: "items", label: "begins" }, { from: "items", to: "empty", label: "no items" },
          { from: "items", to: "payment", label: "one or more items" }, { from: "payment", to: "failure", label: "payment not approved" },
          { from: "payment", to: "shipping", label: "payment approved" }, { from: "shipping", to: "confirm", label: "adjustments complete" },
        ],
      },
      {
        kind: "control_flow",
        nodes: [
          { id: "approved", label: "Approved order" }, { id: "total", label: "Check total", source_ranges: [{ start_line: 11, end_line: 11 }] },
          { id: "zero", label: "Set shipping to 0" }, { id: "twelve", label: "Set shipping to 12" },
          { id: "expedited", label: "Check expedited", source_ranges: [{ start_line: 12, end_line: 12 }] }, { id: "add", label: "Add 20" },
          { id: "keep-fee", label: "Keep current amount" }, { id: "member", label: "Check membership", source_ranges: [{ start_line: 13, end_line: 13 }] },
          { id: "discount", label: "Multiply by 0.9" }, { id: "keep-member", label: "Keep current amount" },
          { id: "return", label: "Return to overview" },
        ],
        edges: [
          { from: "approved", to: "total", label: "starts calculation" }, { from: "total", to: "zero", label: "total > 100" },
          { from: "total", to: "twelve", label: "total ≤ 100" }, { from: "zero", to: "expedited", label: "base set" },
          { from: "twelve", to: "expedited", label: "base set" }, { from: "expedited", to: "add", label: "expedited" },
          { from: "expedited", to: "keep-fee", label: "not expedited" }, { from: "add", to: "member", label: "fee added" },
          { from: "keep-fee", to: "member", label: "no fee added" }, { from: "member", to: "discount", label: "member" },
          { from: "member", to: "keep-member", label: "not a member" }, { from: "discount", to: "return", label: "adjustment complete" },
          { from: "keep-member", to: "return", label: "adjustment complete" },
        ],
      },
    ];
    const combined = {
      kind: "control_flow",
      nodes: visuals.flatMap((visual, index) => visual.nodes.map((node) => ({ ...node, id: `${index}:${node.id}` }))),
      edges: visuals.flatMap((visual, index) => visual.edges.map((edge) => ({ ...edge, from: `${index}:${edge.from}`, to: `${index}:${edge.to}` }))),
    };
    expect(hasCompleteControlFlowCoverage(source, combined)).toBe(true);
  });

  it("rejects a diagram that omits the ternary decision", () => {
    const missingTotalBranch = {
      ...completeVisual,
      nodes: completeVisual.nodes.filter((node) => node.id !== "total"),
      edges: completeVisual.edges.filter((edge) => !["total"].includes(edge.from) && !["total"].includes(edge.to)),
    };
    expect(hasCompleteControlFlowCoverage(shippingSource, missingTotalBranch)).toBe(false);
  });

  it("rejects a decision node that does not identify its source condition", () => {
    const unrelatedDecision = {
      ...completeVisual,
      nodes: completeVisual.nodes.map((node) => node.id === "total" ? { ...node, label: "Check status" } : node),
      edges: completeVisual.edges.map((edge) => edge.from === "total"
        ? { ...edge, label: edge.to === "free" ? "ready" : "not ready" }
        : edge),
    };
    expect(hasCompleteControlFlowCoverage(shippingSource, unrelatedDecision)).toBe(false);
  });

  it("uses source ranges and all terms to match compound conditions", () => {
    const source = "if (order.address && order.approved) return true;";
    const visual = {
      kind: "control_flow",
      nodes: [
        { id: "check", label: "Check address and approved status", source_ranges: [{ start_line: 1, end_line: 1 }] },
        { id: "yes", label: "Continue" },
        { id: "no", label: "Reject" },
      ],
      edges: [
        { from: "check", to: "yes", label: "address and approved" },
        { from: "check", to: "no", label: "otherwise" },
      ],
    };
    expect(analyzeControlFlowCoverage(source, visual).status).toBe("matched");
  });

  it("checks that switch cases and the default path are represented", () => {
    const source = `switch (status) {
  case "ready": return 1;
  case "blocked": throw new Error();
  default: return 0;
}`;
    const visual = {
      kind: "control_flow",
      nodes: [
        { id: "status", label: "Check status", source_ranges: [{ start_line: 1, end_line: 1 }] },
        { id: "ready", label: "Ready" },
        { id: "blocked", label: "Blocked" },
        { id: "default", label: "Default outcome" },
      ],
      edges: [
        { from: "status", to: "ready", label: "ready" },
        { from: "status", to: "blocked", label: "blocked" },
        { from: "status", to: "default", label: "default" },
      ],
    };
    expect(analyzeControlFlowCoverage(source, visual).status).toBe("matched");
    expect(analyzeControlFlowCoverage(source, { ...visual, edges: visual.edges.slice(0, 2) }).status).toBe("incomplete");
  });

  it("reports unsupported language syntax as unverifiable instead of complete", () => {
    const pythonSource = "if not ready:\n    return False\nreturn True";
    expect(analyzeControlFlowCoverage(pythonSource, completeVisual).status).toBe("unverifiable");
    expect(analyzeControlFlowCoverage("while (ready) { work(); }", completeVisual).status).toBe("unverifiable");
  });

  it("reports missing source links as unverifiable", () => {
    const visualWithoutSourceLinks = {
      ...completeVisual,
      nodes: completeVisual.nodes.map(({ id, label }) => ({ id, label })),
    };
    expect(analyzeControlFlowCoverage(shippingSource, visualWithoutSourceLinks).status).toBe("unverifiable");
  });

  it("rejects generic or duplicate outcomes on a decision", () => {
    const vagueBranch = {
      ...completeVisual,
      edges: completeVisual.edges.map((edge) => edge.from === "total" ? { ...edge, label: "either outcome" } : edge),
    };
    expect(hasCompleteControlFlowCoverage(shippingSource, vagueBranch)).toBe(false);
  });

  it("does not apply code-flow checks to concept maps", () => {
    const conceptMap = { ...completeVisual, kind: "concept_map" };
    expect(hasCompleteControlFlowCoverage(shippingSource, conceptMap)).toBe(true);
  });
});

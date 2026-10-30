import { describe, expect, it } from "vitest";
import { findIncompleteCode } from "./incomplete-code";

describe("incomplete code detection", () => {
  it("reports an empty TODO branch and a separate unfinished permission marker", () => {
    const source = [
      "async function fetchUserProfile(userId) {",
      "  if (!userId) return { error: 'Missing user ID' };",
      "  const response = await fetch(`/api/users/${userId}`);",
      "  if (!response.ok) {",
      "    // TODO: handle the error",
      "  }",
      "  const profile = await response.json();",
      "  // TODO: check whether the user is allowed to view this profile",
      "  return { name: profile.name, email: profile.email };",
      "}",
    ].join("\n");

    expect(findIncompleteCode(source)).toEqual([
      {
        kind: "empty_branch",
        title: "Unfinished branch",
        explanation: "This branch has no executable statements and contains an unfinished-work marker. Execution falls through to the next statement; the intended handling is not specified.",
        source_ranges: [{ start_line: 5, end_line: 6 }],
      },
      {
        kind: "unfinished_marker",
        title: "Unfinished-work marker",
        explanation: "The source marks work as unfinished. It does not specify the intended behavior.",
        source_ranges: [{ start_line: 8, end_line: 8 }],
      },
    ]);
  });

  it("does not flag ordinary empty functions or TODO text inside a string", () => {
    expect(findIncompleteCode('function noop() {}\nconst note = "TODO: later";')).toEqual([]);
  });

  it("detects a common hash-comment marker", () => {
    expect(findIncompleteCode("def load():\n    # FIXME: implement this\n    pass")).toHaveLength(1);
  });
});

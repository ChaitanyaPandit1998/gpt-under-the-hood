import { useId, type ReactNode } from "react";

export function VisualDesignerStep({ number, title, optional, id, children }: {
  number: string; title: string; optional?: boolean; id?: string; children: ReactNode;
}) {
  const heading = useId();
  return <section className="visual-designer-stage" aria-labelledby={heading} id={id} tabIndex={id ? -1 : undefined}>
    <h4 id={heading}>
      <span className="visual-designer-stage-number" aria-hidden="true">{number}</span>
      {title}{optional && <small>Optional</small>}
    </h4>
    {children}
  </section>;
}

export const CLOUDFLARE_IMAGE_MODEL = "@cf/black-forest-labs/flux-1-schnell";
export type CloudflareImageCredentials = { accountId: string; apiToken: string };

async function readImageResponse(response: Response): Promise<unknown> {
  const maxBytes = 10_000_000;
  if (Number(response.headers.get("content-length")) > maxBytes || !response.body) throw new Error("Cloudflare returned an empty or oversized image response.");
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let length = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      length += value.length;
      if (length > maxBytes) { await reader.cancel(); throw new Error("Cloudflare returned an oversized image response."); }
      chunks.push(value);
    }
  } finally { reader.releaseLock(); }
  const bytes = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  try { return JSON.parse(new TextDecoder().decode(bytes)); }
  catch { throw new Error("Cloudflare returned invalid JSON."); }
}

export async function generateCloudflareImage(prompt: string, credentials: CloudflareImageCredentials, fetcher: typeof fetch = fetch, signal?: AbortSignal) {
  if (!/^[a-f0-9]{32}$/i.test(credentials.accountId) || !credentials.apiToken.trim()) throw new Error("Configure a valid CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN locally.");
  if (!prompt || prompt.length > 2_048) throw new Error("Invalid image prompt length.");
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 90_000);
  const combined = signal ? AbortSignal.any([signal, controller.signal]) : controller.signal;
  const started = performance.now();
  try {
    combined.throwIfAborted();
    let response: Response;
    try {
      response = await fetcher(`https://api.cloudflare.com/client/v4/accounts/${credentials.accountId}/ai/run/${CLOUDFLARE_IMAGE_MODEL}`, {
        method: "POST", redirect: "error", signal: combined,
        headers: { Authorization: `Bearer ${credentials.apiToken}`, "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, steps: 4 }),
      });
    } catch { throw new Error(combined.aborted ? "Cloudflare image request cancelled or timed out." : "Cloudflare image request failed; credentials and response details were not logged."); }
    if (!response.ok) throw new Error(`Cloudflare image request failed (HTTP ${response.status}). No automatic retries were attempted.`);
    const payload = await readImageResponse(response) as { success?: boolean; result?: { image?: string } };
    if (payload?.success !== true || typeof payload.result?.image !== "string") throw new Error("Cloudflare did not return a successful image result.");
    const encoded = payload.result.image;
    if (!encoded || encoded.length % 4 !== 0 || !/^[a-z0-9+/]*={0,2}$/i.test(encoded)) throw new Error("Cloudflare returned invalid base64 image data.");
    const bytes = Buffer.from(encoded, "base64");
    const extension = bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff ? "jpg"
      : bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])) ? "png" : null;
    if (!extension) throw new Error("Cloudflare returned an unsupported image format.");
    combined.throwIfAborted();
    return { bytes, extension, latencyMs: Math.round(performance.now() - started), byteLength: bytes.length, requestId: response.headers.get("cf-ray") };
  } finally { clearTimeout(timeout); }
}

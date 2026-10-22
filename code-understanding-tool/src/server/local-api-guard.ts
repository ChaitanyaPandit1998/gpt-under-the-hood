export function isTrustedLocalApiRequest(hostHeader: string | undefined, originHeader: string | undefined): boolean {
  if (!hostHeader) return false;

  let requestUrl: URL;
  try {
    requestUrl = new URL(`http://${hostHeader}`);
  } catch {
    return false;
  }

  const hostname = requestUrl.hostname.replace(/^\[|\]$/g, "").toLowerCase();
  const isLoopback = hostname === "localhost" || hostname.endsWith(".localhost") || hostname === "127.0.0.1" || hostname === "::1";
  if (!isLoopback) return false;
  if (!originHeader) return true;

  try {
    return new URL(originHeader).host.toLowerCase() === requestUrl.host.toLowerCase();
  } catch {
    return false;
  }
}

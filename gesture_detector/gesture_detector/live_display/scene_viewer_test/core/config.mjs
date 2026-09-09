const DEFAULT_HAND_HZ = 30;
const MIN_HAND_HZ = 1;
const MAX_HAND_HZ = 120;

export function getHandRateConfig(search) {
  const raw = new URLSearchParams(search).get("hand_hz");
  const requested = raw === null ? DEFAULT_HAND_HZ : Number(raw);
  const valid =
    Number.isFinite(requested) &&
    requested >= MIN_HAND_HZ &&
    requested <= MAX_HAND_HZ;
  const hz = valid ? requested : DEFAULT_HAND_HZ;

  return {
    hz,
    throttleMs: Math.ceil(1000 / hz),
    warning: valid
      ? ""
      : `Invalid hand_hz=${raw}; using ${DEFAULT_HAND_HZ} Hz.`,
  };
}

export function isDemoMode(search) {
  return new URLSearchParams(search).get("demo") === "1";
}

// Supersampling pays for itself on a small embedded card, where CSS downscales
// the extra pixels into crisp thin markers. Past this many CSS pixels it stops
// paying: a maximized canvas at >=2x is several megapixels and reallocating
// that antialiased framebuffer stalls the tab for a second or two.
const SUPERSAMPLE_MAX_CSS_PIXELS = 400_000;
const SUPERSAMPLE_MIN_RATIO = 2;
const SUPERSAMPLE_MAX_RATIO = 3;
const LARGE_CANVAS_MAX_RATIO = 1.5;

export function supersamplePixelRatio(width, height, devicePixelRatio) {
  const ratio = Number.isFinite(devicePixelRatio) && devicePixelRatio > 0
    ? devicePixelRatio
    : 1;
  if (width * height > SUPERSAMPLE_MAX_CSS_PIXELS) {
    return Math.min(ratio, LARGE_CANVAS_MAX_RATIO);
  }
  return Math.min(Math.max(ratio, SUPERSAMPLE_MIN_RATIO), SUPERSAMPLE_MAX_RATIO);
}

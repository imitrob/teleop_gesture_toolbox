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

export function getSelectionStrength(message) {
  const threshold = Number(message?.evidence_threshold);
  const evidence = Number(message?.evidence);
  if (!Number.isFinite(threshold) || threshold <= 0) {
    return 1; // publisher counts no evidence: highlight fully, as it always did
  }
  if (!Number.isFinite(evidence) || evidence <= 0) {
    return 0;
  }
  return Math.min(1, evidence / threshold);
}

export function getSelectedObjectName(message) {
  if (!message || typeof message !== "object") {
    return null;
  }

  if (
    typeof message.target_object_name === "string" &&
    message.target_object_name.trim()
  ) {
    return message.target_object_name.trim();
  }

  const objectNames = Array.isArray(message.object_names)
    ? message.object_names
    : [];
  const targetId = message.target_object_id;
  return Number.isInteger(targetId) &&
    targetId >= 0 &&
    targetId < objectNames.length &&
    typeof objectNames[targetId] === "string" &&
    objectNames[targetId].trim()
    ? objectNames[targetId].trim()
    : null;
}

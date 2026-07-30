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

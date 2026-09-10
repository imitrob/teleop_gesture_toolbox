// Check for the Swipe Detector 3D view geometry in live_display/index.html.
// Run: node tests/test_swipe_plot.js
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const page = fs.readFileSync(
  path.join(__dirname, '..', 'gesture_detector', 'live_display', 'index.html'),
  'utf8');

function slice(from, to){
  const start = page.indexOf(from), end = page.indexOf(to);
  assert.ok(start > 0 && end > start, `${from.trim()} not found in index.html`);
  return page.slice(start, end);
}

const source = slice('  function swipeProject(', '  function swipeStroke(')
             + slice('  function swipeNormalize(', '  function drawSwipePlot(');
const { swipeProject, swipeNormalize } = new Function(
  `let swipeView = {yaw: 0, pitch: 0};
   ${source}
   return {swipeProject: (p, box, view) => { swipeView = view; return swipeProject(p, box); },
           swipeNormalize};`)();

const BOX = {cx: 100, cy: 50, scale: 10};
const FLAT = {yaw: 0, pitch: 0};
const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-9, `${a} != ${b}`);

// Straight on: index 0 is screen right, index 2 is screen up, index 1 is depth.
assert.deepStrictEqual(swipeProject([0, 0, 0], BOX, FLAT), [100, 50]);
assert.deepStrictEqual(swipeProject([1, 0, 0], BOX, FLAT), [110, 50]);
assert.deepStrictEqual(swipeProject([0, 0, 1], BOX, FLAT), [100, 40]);
assert.deepStrictEqual(swipeProject([0, 1, 0], BOX, FLAT), [100, 50]);

// A quarter turn hands the horizontal extent from the right axis to the forward
// one, so a swipe that pointed at the viewer now reads across the plot.
const turned = {yaw: Math.PI / 2, pitch: 0};
close(Math.abs(swipeProject([0, 1, 0], BOX, turned)[0] - BOX.cx), BOX.scale);
close(swipeProject([1, 0, 0], BOX, turned)[0], BOX.cx);

// Pitch tips the depth axis into the vertical, and up stays up.
const tipped = {yaw: 0, pitch: 0.5};
assert.ok(swipeProject([0, 1, 0], BOX, tipped)[1] > 50, 'forward should fall below centre');
assert.ok(swipeProject([0, 0, 1], BOX, tipped)[1] < 50, 'up should stay above centre');

// A path is centred on its middle sample and scaled to unit size, the two steps
// timewarp_lib does before comparing, so only the shape is left.
const swipe = [[0, 0, 0], [0, 0, 50], [0, 0, 100]];
assert.deepStrictEqual(swipeNormalize(swipe), [[0, 0, -1], [0, 0, 0], [0, 0, 1]]);

// Which is what makes a small swipe and a huge one draw as the same path.
assert.deepStrictEqual(
  swipeNormalize([[0, 0, 0], [0, 0, 10], [0, 0, 20]]),
  swipeNormalize([[0, 0, 0], [0, 0, 200], [0, 0, 400]]));

// A resting hand has nothing to scale by: zeros, never a NaN.
const resting = swipeNormalize([[7, 7, 7], [7, 7, 7], [7, 7, 7]]);
assert.deepStrictEqual(resting, [[0, 0, 0], [0, 0, 0], [0, 0, 0]]);
resting.forEach(point => point.forEach(v => assert.ok(Number.isFinite(v))));

// No hand, no path.
assert.deepStrictEqual(swipeNormalize([]), []);

// A resting hand is a few millimetres of sensor noise. Scaled by its own size it
// would fill the plot and read as a moving hand, so the resting displacement is a
// floor on the divisor: it draws small, and a real swipe still fills the plot.
const REST = 0.05; // metres, rest_displacement from the model config
const jitter = [[0, 0, 0], [0.001, 0, -0.002], [-0.001, 0, 0.002]];
const jitterDrawn = swipeNormalize(jitter, REST);
jitterDrawn.forEach(point => point.forEach(v =>
  assert.ok(Math.abs(v) <= 0.2, `resting hand drew at ${v}, expected near the centre`)));

// Without the floor that same noise is a full-size path, which is the confusing part
assert.ok(Math.max(...swipeNormalize(jitter).flat().map(Math.abs)) === 1);

// A swipe at the resting threshold still reaches full size, and a bigger one too
const atThreshold = [[0, 0, -REST/2], [0, 0, 0], [0, 0, REST/2]];
assert.deepStrictEqual(swipeNormalize(atThreshold, REST), [[0, 0, -1], [0, 0, 0], [0, 0, 1]]);
assert.deepStrictEqual(
  swipeNormalize([[0, 0, -1], [0, 0, 0], [0, 0, 1]], REST),
  [[0, 0, -1], [0, 0, 0], [0, 0, 1]]);

// The offset the hand happens to sit at never shows: only its shape does.
assert.deepStrictEqual(
  swipeNormalize([[100, 0, 0], [100, 0, 50], [100, 0, 100]]),
  swipeNormalize(swipe));

console.log('ok test_swipe_plot');

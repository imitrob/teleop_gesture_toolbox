// Check for the STT card word parser in live_display/index.html.
// Run: node tests/test_stt_card.js
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const page = fs.readFileSync(
  path.join(__dirname, '..', 'gesture_detector', 'live_display', 'index.html'),
  'utf8');
const start = page.indexOf('  function sttProbability(');
const end = page.indexOf('  function sttEmptyNode(');
assert.ok(start > 0 && end > start, 'sttWords() not found in index.html');
const sttWords = new Function(`${page.slice(start, end)}; return sttWords;`)();

// Alternatives sorted by likelihood, the selected word dropped from them.
const words = sttWords('', JSON.stringify([
  {word: ' pick ', alts: {pick: 0.7, pink: 0.1, kick: 0.2}},
]));
assert.deepStrictEqual(words, [{
  word: 'pick',
  alternatives: [{word: 'kick', probability: 0.2}, {word: 'pink', probability: 0.1}],
}]);

// Malformed or absent metadata falls back to the plain transcript.
for (const broken of ['', 'not json', '{}', JSON.stringify([{word: ''}])]) {
  assert.deepStrictEqual(sttWords('pick   the cube', broken), [
    {word: 'pick', alternatives: []},
    {word: 'the', alternatives: []},
    {word: 'cube', alternatives: []},
  ], `fallback failed for ${JSON.stringify(broken)}`);
}

// Nothing to say: no words, so the card shows "Waiting for speech…".
assert.deepStrictEqual(sttWords('', ''), []);
assert.deepStrictEqual(sttWords(null, null), []);

// Out-of-range and non-numeric likelihoods are clamped, never NaN.
assert.deepStrictEqual(
  sttWords('', JSON.stringify([{word: 'go', alts: {up: 3, down: -1, left: 'x'}}])),
  [{word: 'go', alternatives: [
    {word: 'up', probability: 1},
    {word: 'down', probability: 0},
    {word: 'left', probability: 0},
  ]}]);

console.log('ok');

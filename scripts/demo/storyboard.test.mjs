import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  BEATS, GEOMETRY, DRILL_IN_ROW,
  beatsForCut, dwellFor, captionFor, totalDuration, validateStoryboard,
} from './storyboard.mjs';

const ACTION_TYPES = new Set(['scrollTo', 'click', 'select', 'hover', 'fill', 'ring', 'unring', 'pause']);
const ASSERT_TYPES = new Set(['domContains', 'visible', 'attr', 'minChildren', 'textMatches']);

test('the storyboard validates clean', () => {
  assert.deepEqual(validateStoryboard(), []);
});

test('geometry is the single recording geometry for both cuts', () => {
  assert.deepEqual(GEOMETRY, { width: 1440, height: 810 });
});

test('there are 13 beats with unique ids', () => {
  assert.equal(BEATS.length, 13);
  assert.equal(new Set(BEATS.map((b) => b.id)).size, 13);
});

test('every beat declares known action and assert types', () => {
  for (const beat of BEATS) {
    assert.ok(beat.actions.length > 0 || beat.id === 'cold-open', `${beat.id} has no actions`);
    for (const a of beat.actions) {
      assert.ok(ACTION_TYPES.has(a.type), `${beat.id}: bad action ${a.type}`);
    }
    assert.ok(beat.asserts.length > 0, `${beat.id} has no assertions`);
    for (const a of beat.asserts) {
      assert.ok(ASSERT_TYPES.has(a.type), `${beat.id}: bad assert ${a.type}`);
    }
  }
});

test('every textMatches pattern compiles', () => {
  for (const beat of BEATS) {
    for (const a of beat.asserts.filter((x) => x.type === 'textMatches')) {
      assert.doesNotThrow(() => new RegExp(a.pattern), `${beat.id}: ${a.pattern}`);
    }
  }
});

test('the gif cut keeps exactly the seven marked beats, in order', () => {
  assert.deepEqual(beatsForCut('gif').map((b) => b.id), [
    'cold-open', 'cost-tiles', 'board', 'drill-writeup', 'drill-proposals', 'radial-tree', 'close',
  ]);
});

test('the full cut keeps every beat, in declaration order', () => {
  assert.deepEqual(beatsForCut('full').map((b) => b.id), BEATS.map((b) => b.id));
});

test('cut durations match the spec budgets', () => {
  assert.equal(totalDuration('full'), 62.0);
  assert.equal(totalDuration('gif'), 16.5);
});

test('every gif beat carries its own shorter caption and dwell', () => {
  for (const beat of beatsForCut('gif')) {
    assert.equal(typeof beat.caption.gif, 'string', `${beat.id} has no gif caption`);
    assert.ok(beat.caption.gif.length > 0, `${beat.id} gif caption is empty`);
    assert.ok(
      beat.caption.gif.length <= beat.caption.full.length,
      `${beat.id} gif caption is longer than the full one`,
    );
    assert.ok(dwellFor(beat, 'gif') <= dwellFor(beat, 'full'), `${beat.id} gif dwell exceeds full`);
  }
});

test('dwellFor and captionFor fall back to the full cut', () => {
  const ask = BEATS.find((b) => b.id === 'the-ask');
  assert.equal(ask.cuts.includes('gif'), false);
  assert.equal(dwellFor(ask, 'full'), 4.0);
  assert.equal(captionFor(ask, 'full'), ask.caption.full);
});

test('the drill-in beats target the rank-17 candidate', () => {
  assert.equal(DRILL_IN_ROW, 'tbody#cand-vllm_v1_kv_offload-0009');
  const writeup = BEATS.find((b) => b.id === 'drill-writeup');
  assert.ok(writeup.actions.some((a) => a.type === 'click' && a.sel.startsWith(DRILL_IN_ROW)));
});

test('no beat ever touches the theme toggle', () => {
  for (const beat of BEATS) {
    for (const a of beat.actions) {
      assert.notEqual(a.sel, '#theme', `${beat.id} clicks the theme toggle`);
    }
  }
});

test('beatsForCut rejects an unknown cut', () => {
  assert.throws(() => beatsForCut('nope'), /unknown cut/);
});

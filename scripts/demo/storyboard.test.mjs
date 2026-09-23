import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  BEATS, GEOMETRY, DRILL_IN_ROW, DOWNSTREAM_SECTION, TREE_VIEWPORT_COVERAGE,
  FINDINGS_SECTION, TOP_FINDING,
  beatsForCut, dwellFor, captionFor, totalDuration, validateStoryboard,
} from './storyboard.mjs';

const ACTION_TYPES = new Set([
  'scrollTo', 'scrollCenter', 'click', 'select', 'hover', 'fill', 'ring', 'unring', 'pause',
]);
const ASSERT_TYPES = new Set([
  'domContains', 'visible', 'attr', 'minChildren', 'textMatches', 'inViewport', 'viewportCoverage',
]);

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
    assert.ok(beat.actions.length > 0, `${beat.id} has no actions`);
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

test('the gif cut keeps exactly the eight marked beats, in order', () => {
  // Eight, not seven: the findings catalogue is in the short cut now, and it sits
  // between the candidate that cites papers and the tree.
  assert.deepEqual(beatsForCut('gif').map((b) => b.id), [
    'cost-tiles', 'leaderboard-reveal', 'board', 'drill-writeup', 'drill-proposals',
    'findings', 'radial-tree', 'close',
  ]);
});

test('the demo opens on the overview, then reveals the leaderboard', () => {
  // The order of the shared array IS the play order, so this pins both cuts at once.
  assert.deepEqual(BEATS.slice(0, 3).map((b) => b.id), ['the-ask', 'cost-tiles', 'leaderboard-reveal']);
  // The objective beat is full-only and sits against the overview, so the long cut
  // states the objective first and the overview still opens both cuts' picture.
  const ask = BEATS.find((b) => b.id === 'the-ask');
  assert.deepEqual(ask.cuts, ['full']);
  const ids = BEATS.map((b) => b.id);
  assert.equal(Math.abs(ids.indexOf('the-ask') - ids.indexOf('cost-tiles')), 1);
  // Whichever cut, the first thing shown is the overview.
  assert.equal(beatsForCut('gif')[0].id, 'cost-tiles');
  assert.equal(beatsForCut('full')[1].id, 'cost-tiles');
  // No beat is named for a cold open any more, because nothing is pre-positioned.
  assert.equal(ids.includes('cold-open'), false);
});

test('the leaderboard beat scrolls itself to the board', () => {
  // It used to be first and to rely on record.mjs parking the page on the board before
  // the loop; second-or-later, it has to get there on its own.
  const reveal = BEATS.find((b) => b.id === 'leaderboard-reveal');
  assert.deepEqual(reveal.cuts, ['gif', 'full']);
  assert.deepEqual(reveal.actions, [{ type: 'scrollTo', sel: '#lbtable' }]);
  assert.ok(reveal.asserts.some((a) => a.type === 'visible' && a.sel === '#lbtable'));
  // Its caption quotes 173, so the beat asserts the board's own count.
  assert.ok(
    reveal.asserts.some((a) => a.type === 'textMatches' && a.sel === '.sechead h2' && /173/.test(a.pattern)),
    'the leaderboard beat must assert the 173 its caption quotes',
  );
});

test('the overview caption leads with the candidate count in both cuts', () => {
  const tiles = BEATS.find((b) => b.id === 'cost-tiles');
  for (const cut of ['gif', 'full']) {
    const caption = captionFor(tiles, cut);
    assert.match(caption, /^<b>173<\/b> candidates · /, `${cut} caption must lead with 173 candidates`);
    assert.match(caption, /<b>544<\/b> proposals/, `${cut} caption keeps the proposal count`);
  }
  assert.ok(captionFor(tiles, 'full').includes('$35.75'));
});

test('the overview beat asserts 173 against the candidate figure, not a loose 173', () => {
  const tiles = BEATS.find((b) => b.id === 'cost-tiles');
  const fig = tiles.asserts.find((a) => a.type === 'textMatches' && a.sel === '.hero .fig');
  assert.ok(fig, 'the candidate count needs a scoped assertion');
  assert.equal(fig.pattern, '^173$');
  // Anchored, so it cannot be satisfied by a number that merely contains 173.
  assert.match('173', new RegExp(fig.pattern));
  assert.doesNotMatch('1730', new RegExp(fig.pattern));
  // The label is asserted with the number, so the figure cannot silently become
  // something else that happens to read 173.
  assert.ok(tiles.asserts.some(
    (a) => a.type === 'textMatches' && a.sel === '.hero .figlab' && /candidates/.test(a.pattern),
  ));
  // "173" appears many times on this page -- the Awaiting measurement tile is 173 too --
  // so a document-wide substring check would not witness the candidate count at all.
  assert.deepEqual(tiles.asserts.filter((a) => a.type === 'domContains' && a.text.includes('173')), []);
});

test('the tree beat scrolls the tree into frame and guards how much of it shows', () => {
  const tree = BEATS.find((b) => b.id === 'radial-tree');
  const types = tree.actions.map((a) => a.type);
  // The toggle is clicked, the tree is given time to draw, and only then is it framed.
  assert.ok(types.indexOf('scrollCenter') > types.lastIndexOf('click'), 'frame the tree after toggling it');
  const centre = tree.actions.find((a) => a.type === 'scrollCenter');
  assert.equal(centre.sel, '#rtplot svg');
  // Framed after the toggle but before the beat's closing pause, so the centred tree is
  // held on screen rather than glimpsed as the beat hands over.
  const centreAt = tree.actions.indexOf(centre);
  assert.ok(centreAt < tree.actions.length - 1, 'the framing must not be the beat\'s last act');
  assert.deepEqual(tree.actions.slice(centreAt + 1), [{ type: 'pause', ms: 1400 }]);

  // Hot mode is on from the board beat and the tree obeys it, so the beat clears it:
  // its caption promises all 173 and the hub is now the middle of the frame.
  assert.ok(tree.actions.some((a) => a.type === 'click' && a.sel === '#hotbtn'));
  assert.ok(
    tree.asserts.some((a) => a.type === 'textMatches' && a.sel === '#count' && a.pattern === '^173 / 173 shown$'),
    'the tree beat must prove nothing is filtering the 173 its caption promises',
  );

  const cover = tree.asserts.find((a) => a.type === 'viewportCoverage');
  assert.ok(cover, 'the tree beat must assert how much of the frame the tree fills');
  assert.equal(cover.sel, centre.sel, 'the assertion must watch what the beat scrolls');
  assert.equal(cover.minFraction, TREE_VIEWPORT_COVERAGE);
  // inViewport is the wrong tool here: the tree is taller than the viewport, so no
  // scroll position can put its whole box inside the frame.
  assert.deepEqual(tree.asserts.filter((a) => a.type === 'inViewport'), []);
});

test('the tree coverage threshold is a real fraction, clear of the old framing', () => {
  assert.equal(typeof TREE_VIEWPORT_COVERAGE, 'number');
  assert.ok(TREE_VIEWPORT_COVERAGE > 0 && TREE_VIEWPORT_COVERAGE <= 1);
  // The toggle-only scroll this replaced left 307 of 810px of the plot in frame (0.38).
  assert.ok(TREE_VIEWPORT_COVERAGE > 307 / 810, 'the threshold must reject the pre-scroll framing');
});

test('the full cut keeps every beat, in declaration order', () => {
  assert.deepEqual(beatsForCut('full').map((b) => b.id), BEATS.map((b) => b.id));
});

test('cut durations match the spec budgets', () => {
  assert.equal(totalDuration('full'), 62.0);
  // 20.5s, not 16.5s: the findings beat joined the short cut and brought a 4.0s dwell
  // with it. The full cut is unchanged, because that beat was always in it at 4.0s.
  assert.equal(totalDuration('gif'), 20.5);
  // The arithmetic, spelled out: the gif cut is the old 16.5s plus that one dwell. It
  // lives here rather than in its own test so the gif budget is asserted in exactly one
  // place -- the throwaway nortree variant re-keys this number, and a second copy would
  // widen what that patch breaks.
  const findings = BEATS.find((b) => b.id === 'findings');
  assert.equal(totalDuration('gif') - dwellFor(findings, 'gif'), 16.5);
});

test('the findings beat kept its full dwell, so the full budget did not move', () => {
  const findings = BEATS.find((b) => b.id === 'findings');
  assert.equal(dwellFor(findings, 'gif'), 4.0);
  assert.equal(dwellFor(findings, 'full'), 4.0);
  // The full cut cannot have moved, because the beat only changed cuts and position.
  const fullWithout = beatsForCut('full')
    .filter((b) => b.id !== 'findings')
    .reduce((s, b) => s + dwellFor(b, 'full'), 0);
  assert.equal(Math.round((fullWithout + 4.0) * 10) / 10, 62.0);
});

test('the findings catalogue is in both cuts, right after the candidate that cites papers', () => {
  const findings = BEATS.find((b) => b.id === 'findings');
  assert.ok(findings, 'the findings beat must exist');
  assert.deepEqual(findings.cuts, ['gif', 'full']);
  const ids = BEATS.map((b) => b.id);
  // Immediately after drill-proposals in the shared array, which is the play order for
  // both cuts at once -- so the catalogue answers "which papers?" in the long cut too.
  assert.equal(ids[ids.indexOf('drill-proposals') + 1], 'findings');
  // It used to sit before the board, upstream of the drill-in. It must not any more.
  assert.ok(ids.indexOf('findings') > ids.indexOf('board'));
  const gif = beatsForCut('gif').map((b) => b.id);
  assert.equal(gif[gif.indexOf('drill-proposals') + 1], 'findings');
  const full = beatsForCut('full').map((b) => b.id);
  assert.equal(full[full.indexOf('drill-proposals') + 1], 'findings');
  // The beat it hands over to, so its filters must not break that one: asserted on the
  // shared array, which is the play order, and not on one cut's membership.
  assert.equal(ids[ids.indexOf('findings') + 1], 'radial-tree');
});

test('the findings beat asks the payoff question: filter to papers, then rank by proposals', () => {
  const findings = BEATS.find((b) => b.id === 'findings');
  const selects = findings.actions.filter((a) => a.type === 'select');
  assert.deepEqual(selects, [
    { type: 'select', sel: '#fsrc', value: 'paper' },
    { type: 'select', sel: '#fsort', value: 'props:-1' },
  ], 'the source filter must land before the sort, so the frame ends on ranked papers');
  // The sort is the last thing the beat changes, so the closing frame is the ranked one.
  const types = findings.actions.map((a) => a.type);
  assert.equal(types.lastIndexOf('select'), findings.actions.indexOf(selects[1]));
  assert.deepEqual(findings.actions[findings.actions.length - 1], { type: 'pause', ms: 600 });
  // It no longer types into the search box: `actions` is shared by both cuts and the
  // fill-pause-clear does not fit the gif's dwell.
  assert.deepEqual(findings.actions.filter((a) => a.type === 'fill'), []);
  // It must not touch the catalogue's other two controls, whose figures it does not quote.
  for (const a of findings.actions) {
    assert.notEqual(a.sel, '#fmod', 'the beat quotes no per-module figure');
    assert.notEqual(a.sel, '#fuse', 'the beat quotes no used/unused figure');
  }
});

test('both findings captions carry exactly the figures the beat proves', () => {
  const findings = BEATS.find((b) => b.id === 'findings');
  for (const cut of ['gif', 'full']) {
    const caption = captionFor(findings, cut);
    assert.match(caption, /<b>97<\/b> findings/, `${cut}: the catalogue's own total`);
    assert.match(caption, /<b>46<\/b> sites/, `${cut}: where the findings were published`);
    assert.match(caption, /<b>27<\/b> from papers/, `${cut}: the filtered count`);
    assert.match(caption, /<b>6<\/b> proposals|produced <b>6<\/b>/, `${cut}: the top row's payoff`);
  }
  // The gif caption is the short form of the same claim.
  assert.ok(captionFor(findings, 'gif').length < captionFor(findings, 'full').length);
  assert.match(captionFor(findings, 'full'), /whether it paid off/);
});

test('the findings beat witnesses each caption figure through a scoped selector', () => {
  const findings = BEATS.find((b) => b.id === 'findings');
  const text = (sel) => findings.asserts.find((a) => a.type === 'textMatches' && a.sel === sel);

  // 27 of 97, off the catalogue's own counter, anchored end to end.
  const count = text('#fcount');
  assert.ok(count, 'the filtered count needs an assertion on #fcount');
  assert.equal(count.pattern, '^27 / 97 shown$');
  assert.match('27 / 97 shown', new RegExp(count.pattern));
  assert.doesNotMatch('127 / 97 shown', new RegExp(count.pattern));
  assert.doesNotMatch('97 / 97 shown', new RegExp(count.pattern));

  // The section's own 97, from the section's own heading.
  const heading = text(`${FINDINGS_SECTION} > h2`);
  assert.ok(heading, "the section's 97 needs an assertion on the section heading");
  assert.match(heading.pattern, /97 findings/);
  assert.ok(FINDINGS_SECTION.includes('#fq'), 'the section is addressed by the control it owns');

  // 46 sites, from the label the sort control states it under.
  const sites = text('#fsort option[value="host:1"]');
  assert.ok(sites, 'the 46 sites needs an assertion on the sort option that states it');
  assert.match(sites.pattern, /46 sites/);

  // The 97, the 46 and the 27 are never taken from a document-wide substring check: the
  // page repeats all three, so such a check would pass with the catalogue gone.
  assert.deepEqual(findings.asserts.filter((a) => a.type === 'domContains'), []);
});

test('the findings beat proves the top row after the sort, not merely that some row matches', () => {
  const findings = BEATS.find((b) => b.id === 'findings');
  // Pinned to one element: :visible skips the filtered-out tbodies the sort reorders
  // anyway, and nth=0 is what makes it the top row rather than any row.
  assert.ok(TOP_FINDING.includes(':visible'));
  assert.match(TOP_FINDING, /nth=0$/);

  const attrs = findings.asserts.filter((a) => a.type === 'attr' && a.sel === TOP_FINDING);
  assert.deepEqual(attrs, [
    { type: 'attr', sel: TOP_FINDING, name: 'data-src', equals: 'paper' },
    { type: 'attr', sel: TOP_FINDING, name: 'data-props', equals: '6' },
  ], 'the top row must be proved a paper finding with the caption\'s proposal count');

  // And the two cells a viewer actually reads off the frame.
  const badge = findings.asserts.find(
    (a) => a.type === 'textMatches' && a.sel === `${TOP_FINDING} >> tr.frow .tag`,
  );
  assert.ok(badge, 'the rendered source badge needs an assertion');
  assert.equal(badge.pattern, '^paper$');
  const props = findings.asserts.find(
    (a) => a.type === 'textMatches' && a.sel === `${TOP_FINDING} >> tr.frow td.r >> nth=1`,
  );
  assert.ok(props, 'the rendered Proposals cell needs an assertion');
  assert.equal(props.pattern, '^6$');
  // Anchored, so a cell reading 16 or 61 cannot satisfy the caption's 6.
  assert.doesNotMatch('16', new RegExp(props.pattern));
  assert.doesNotMatch('61', new RegExp(props.pattern));

  // Every assertion in the beat is scoped to the catalogue or to its top row.
  for (const a of findings.asserts) {
    assert.ok(
      a.sel.startsWith(TOP_FINDING) || a.sel.startsWith(FINDINGS_SECTION)
        || ['#fcount', '#fsort option[value="host:1"]'].includes(a.sel),
      `unscoped assertion selector: ${a.sel}`,
    );
  }
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

test('the close beat looks at the downstream artifacts it narrates', () => {
  const close = BEATS.find((b) => b.id === 'close');
  // Both cuts end on this beat, so retargeting its one scroll retargets both.
  assert.deepEqual(close.cuts, ['gif', 'full']);
  assert.deepEqual(close.actions, [{ type: 'scrollTo', sel: DOWNSTREAM_SECTION }]);
  assert.ok(
    DOWNSTREAM_SECTION.includes('Downstream artifacts'),
    'the close beat must scroll to the downstream artifacts section, not the leaderboard',
  );
  // It must no longer share the radial-tree beat's target, which is why it never moved.
  const tree = BEATS.find((b) => b.id === 'radial-tree');
  const treeScrolls = tree.actions.filter((a) => a.type === 'scrollTo').map((a) => a.sel);
  assert.ok(treeScrolls.includes('#lbview'));
  for (const a of close.actions) assert.notEqual(a.sel, '#lbview');

  // A domContains cannot witness scroll position; the beat needs an assertion that can.
  const inView = close.asserts.filter((a) => a.type === 'inViewport');
  assert.equal(inView.length, 1);
  assert.equal(inView[0].sel, DOWNSTREAM_SECTION);
  assert.equal(inView[0].sel, close.actions[0].sel, 'the assertion must watch what the beat scrolls to');
  // The figure guards stay.
  assert.deepEqual(
    close.asserts.filter((a) => a.type === 'domContains').map((a) => a.text),
    ['Awaiting measurement', 'no candidate has a measured speedup yet'],
  );
});

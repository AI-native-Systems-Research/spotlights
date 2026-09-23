import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  BEATS, GEOMETRY, DRILL_IN_ROW, DRILL_IN_CAND, DOWNSTREAM_SECTION, TREE_VIEWPORT_COVERAGE,
  FINDINGS_SECTION, TOP_FINDING, TOP_FINDING_FIRST_PROP, TOP_FINDING_CAND_LINK,
  LEADERBOARD_HEAD, PROPOSALS_VIEWPORT_COVERAGE, ALIGN_BLOCKS,
  STICKY_CLEARANCE_LB, STICKY_CLEARANCE_FND, CLOSE_CLEARANCE, ALIGN_TOLERANCE_PX,
  beatsForCut, dwellFor, captionFor, totalDuration, validateStoryboard, beatProblems,
} from './storyboard.mjs';

const ACTION_TYPES = new Set([
  // scrollCenter became scrollAlign, which takes the alignment as an option rather than
  // spelling one per action type; scrollStepped reaches the same place in visible hops.
  'scrollTo', 'scrollAlign', 'scrollStepped', 'click', 'select', 'hover', 'fill',
  'ring', 'unring', 'pause',
]);
/**
 * `beatProblems` run over a beat that is valid except for what the caller overrides, so a
 * rule can be tested by breaking it. The defaults are the minimum a beat needs to be
 * otherwise clean, which is why a returned problem is always about the override.
 */
function withBeat(overrides) {
  return beatProblems({
    id: 'probe',
    cuts: ['full'],
    dwell: { full: 1.0 },
    caption: { full: 'probe' },
    actions: [],
    asserts: [{ type: 'visible', sel: '#x' }],
    ...overrides,
  });
}

const ASSERT_TYPES = new Set([
  'domContains', 'visible', 'attr', 'minChildren', 'textMatches', 'inViewport', 'viewportCoverage',
  // Both added because inViewport passes on the two framings that shipped broken: content
  // under a sticky header, and a short section stranded at the bottom of the frame.
  'unoccluded', 'nearTop',
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
  // Eight, not seven: the findings catalogue is in the short cut too. It now follows the
  // tree rather than preceding it -- the whole run, then where its evidence came from.
  assert.deepEqual(beatsForCut('gif').map((b) => b.id), [
    'cost-tiles', 'leaderboard-reveal', 'board', 'drill-writeup', 'drill-proposals',
    'radial-tree', 'findings', 'close',
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

test('the leaderboard beat frames the section heading above the first rows', () => {
  // It used to be first and to rely on record.mjs parking the page on the board before
  // the loop; second-or-later, it has to get there on its own -- and scrollIntoViewIfNeeded
  // on the table put the heading above the top of the frame, so the shot was an untitled
  // list. It now aligns the heading itself to the top of the viewport.
  const reveal = BEATS.find((b) => b.id === 'leaderboard-reveal');
  assert.deepEqual(reveal.cuts, ['gif', 'full']);
  assert.deepEqual(reveal.actions, [
    { type: 'scrollAlign', sel: LEADERBOARD_HEAD, block: 'start' },
  ]);
  assert.ok(reveal.asserts.some((a) => a.type === 'visible' && a.sel === '#lbtable'));
  // Its caption quotes 173, so the beat asserts the board's own count -- and now the
  // whole heading, anchored, because the heading is what the viewer reads off the frame.
  const head = reveal.asserts.find(
    (a) => a.type === 'textMatches' && a.sel === `${LEADERBOARD_HEAD} h2`,
  );
  assert.ok(head, 'the leaderboard beat must assert the 173 its caption quotes');
  assert.match(head.pattern, /173/);
  assert.equal(head.pattern, '^Candidate leaderboard \u2014 173 ranked$');
  assert.match('Candidate leaderboard \u2014 173 ranked', new RegExp(head.pattern));
  assert.doesNotMatch('Candidate leaderboard \u2014 173 ranked (filtered)', new RegExp(head.pattern));

  // Reading right is not being on screen, and being on screen is the point of the change.
  const inView = reveal.asserts.filter((a) => a.type === 'inViewport').map((a) => a.sel);
  assert.deepEqual(inView, [`${LEADERBOARD_HEAD} h2`, 'table.lb tbody.cand:visible >> nth=0']);
  // The heading the beat scrolls to is the heading it proves is in frame.
  assert.ok(inView[0].startsWith(reveal.actions[0].sel));
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
  assert.ok(types.indexOf('scrollAlign') > types.lastIndexOf('click'), 'frame the tree after toggling it');
  const centre = tree.actions.find((a) => a.type === 'scrollAlign');
  assert.equal(centre.sel, '#rtplot svg');
  // The alignment is now an option on one scroll action rather than its own action type,
  // so the beat has to say which alignment it wants: centred, for something oversized.
  assert.equal(centre.block, 'center');
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
  // 64.0, not 62.0: the findings beat's dwell went 4.0 -> 6.0 to pay for opening the top
  // paper finding and re-framing it so the candidate link is legible.
  assert.equal(totalDuration('full'), 64.0);
  // 23.0, not 20.5: that same +2.0, plus the +0.5 the stepped scroll costs drill-proposals
  // in the short cut. Nothing else moved.
  assert.equal(totalDuration('gif'), 23.0);
  // The arithmetic, spelled out: the gif cut without the findings beat is 17.0s -- the old
  // 16.5 plus drill-proposals' half second. It lives here rather than in its own test so
  // the gif budget is asserted in exactly one place -- the throwaway nortree variant
  // re-keys this number, and a second copy would widen what that patch breaks.
  const findings = BEATS.find((b) => b.id === 'findings');
  assert.equal(totalDuration('gif') - dwellFor(findings, 'gif'), 17.0);
  const props = BEATS.find((b) => b.id === 'drill-proposals');
  assert.equal(
    Math.round((totalDuration('gif') - dwellFor(findings, 'gif') - dwellFor(props, 'gif')) * 10) / 10,
    13.5,
  );
});

test('the findings dwell pays for the choreography it grew, in both cuts alike', () => {
  const findings = BEATS.find((b) => b.id === 'findings');
  // 6.0, not 4.0: the beat now clicks the top finding open and re-frames it, which spends
  // ~5.3s of the dwell-as-deadline budget before the caption's fade-out reserve. Both cuts
  // get the same number because `actions` is shared -- the work is identical in each.
  assert.equal(dwellFor(findings, 'gif'), 6.0);
  assert.equal(dwellFor(findings, 'full'), 6.0);
  // The +2.0 is the *only* reason the full budget moved: every other full dwell is as it
  // was, so the full cut is the old 62.0 with this beat's two extra seconds in it.
  const fullWithout = beatsForCut('full')
    .filter((b) => b.id !== 'findings')
    .reduce((s, b) => s + dwellFor(b, 'full'), 0);
  assert.equal(Math.round((fullWithout + 4.0) * 10) / 10, 62.0);
  assert.equal(Math.round((fullWithout + dwellFor(findings, 'full')) * 10) / 10, 64.0);
});

test('the findings catalogue is in both cuts, and is the last thing shown before the close', () => {
  const findings = BEATS.find((b) => b.id === 'findings');
  assert.ok(findings, 'the findings beat must exist');
  assert.deepEqual(findings.cuts, ['gif', 'full']);
  const ids = BEATS.map((b) => b.id);
  // It moved from just after drill-proposals to just after radial-tree, so the sequence
  // is: this candidate, then the whole run as a tree, then where the evidence came from
  // and which candidate it produced. Asserted on the shared array, which IS the play
  // order, so this pins both cuts at once.
  assert.equal(ids[ids.indexOf('radial-tree') + 1], 'findings');
  assert.equal(ids[ids.indexOf('drill-proposals') + 1], 'radial-tree');
  assert.deepEqual(ids.slice(-4), ['drill-proposals', 'radial-tree', 'findings', 'close']);
  // It used to sit before the board, upstream of the drill-in. It must not any more: the
  // beat now points at the candidate the drill-in opened, which has to have happened.
  assert.ok(ids.indexOf('findings') > ids.indexOf('board'));
  assert.ok(ids.indexOf('findings') > ids.indexOf('drill-proposals'));
  for (const cut of ['gif', 'full']) {
    const order = beatsForCut(cut).map((b) => b.id);
    assert.equal(order[order.indexOf('radial-tree') + 1], 'findings');
    assert.equal(order[order.indexOf('findings') + 1], 'close');
  }
  // The beat it hands over to, so its filters must not break that one.
  assert.equal(ids[ids.indexOf('findings') + 1], 'close');
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
  // Not scrollTo. scrollIntoViewIfNeeded does nothing while any part of the target is
  // already in frame, and the downstream section sits directly below the findings
  // catalogue -- so from the moment the findings beat moved in front of this one, the
  // closing shot never moved at all.
  assert.deepEqual(close.actions, [{
    type: 'scrollAlign', sel: DOWNSTREAM_SECTION, block: 'start', marginTop: CLOSE_CLEARANCE,
  }]);
  assert.deepEqual(close.actions.filter((a) => a.type === 'scrollTo'), []);
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

  // inViewport alone is what let the stranded framing through: the section is 129px tall,
  // so it fitted from y 681..810 -- the bottom strip of a frame that was otherwise the
  // previous beat -- with a fraction of a pixel to spare. Only a bound on where its top
  // edge landed tells the two frames apart, and it has to be tight enough to reject the
  // bottom of the frame by a wide margin.
  const anchored = close.asserts.filter((a) => a.type === 'nearTop');
  assert.equal(anchored.length, 1);
  assert.equal(anchored[0].sel, DOWNSTREAM_SECTION);
  assert.equal(anchored[0].maxY, CLOSE_CLEARANCE + ALIGN_TOLERANCE_PX);
  assert.ok(anchored[0].maxY < GEOMETRY.height / 8, 'a loose bound would re-admit the defect');
  // The figure guards stay.
  assert.deepEqual(
    close.asserts.filter((a) => a.type === 'domContains').map((a) => a.text),
    ['Awaiting measurement', 'no candidate has a measured speedup yet'],
  );
});

test('the findings beat opens the top paper finding rather than stopping at the sort', () => {
  const findings = BEATS.find((b) => b.id === 'findings');
  const types = findings.actions.map((a) => a.type);
  // The click comes after both selects, so what is opened is the top row of the *sorted,
  // filtered* table and not whatever happened to be first beforehand.
  const clickAt = types.indexOf('click');
  assert.ok(clickAt > types.lastIndexOf('select'), 'open the top row after the sort lands');
  const open = findings.actions[clickAt];
  // The title cell, not the row: the row's centre is the outbound source link, which is
  // target=_blank and which the section deliberately does not treat as the expander.
  assert.equal(open.sel, `${TOP_FINDING} >> tr.frow .ftitle`);
  assert.ok(open.sel.startsWith(TOP_FINDING), 'the top row is what gets opened');

  // The detail unfolds ~700px below a row two thirds down the frame, so the beat re-frames
  // after opening it -- and re-frames the whole row, because aligning the detail's own
  // body hides the two headings under the table's sticky header.
  const frameAt = types.indexOf('scrollAlign');
  assert.ok(frameAt > clickAt, 'frame the detail after it has been opened');
  assert.deepEqual(findings.actions[frameAt], {
    type: 'scrollAlign', sel: TOP_FINDING, block: 'start', marginTop: STICKY_CLEARANCE_FND,
  });
  // Framed before the closing pause, so the opened finding is held rather than glimpsed.
  assert.deepEqual(findings.actions.slice(frameAt + 1), [{ type: 'pause', ms: 600 }]);

  // The detail is visible -- the one assertion that fails if the click missed the
  // expander, since tr.fdetail is in the DOM either way.
  assert.ok(findings.asserts.some(
    (a) => a.type === 'visible' && a.sel === `${TOP_FINDING} >> tr.fdetail`,
  ), 'the expanded detail must be proved visible, not merely present');

  // What the paper taught, and the passage it was taught from.
  const headings = findings.asserts
    .filter((a) => a.type === 'textMatches' && a.sel === `${TOP_FINDING} >> tr.fdetail h4`)
    .map((a) => a.pattern);
  assert.deepEqual(headings, ['^Technique extracted$', '^Supporting evidence$']);

  // The caption's "6 proposals" as the frame renders it, word for word.
  const props = findings.asserts.find(
    (a) => a.type === 'textMatches' && a.sel === `${TOP_FINDING} >> .fprops > h4`,
  );
  assert.ok(props, 'the proposal-count heading needs its own assertion');
  assert.equal(props.pattern, '^6 proposals it produced$');
  assert.match('6 proposals it produced', new RegExp(props.pattern));
  assert.doesNotMatch('16 proposals it produced', new RegExp(props.pattern));
  // The caption claims that heading's number, so the two cannot disagree.
  for (const cut of ['gif', 'full']) {
    assert.match(captionFor(findings, cut), /<b>6<\/b> proposals/, `${cut}: the heading's 6`);
  }

  // Every one of those new assertions is scoped to this row, not to the catalogue at
  // large: the page renders 27 filtered findings, each with its own detail and headings.
  for (const a of findings.asserts.filter((x) => /fdetail|fprops|pitem|data-cand/.test(x.sel))) {
    assert.ok(a.sel.startsWith(TOP_FINDING), `unpinned selector: ${a.sel}`);
  }
});

test('the findings beat proves the top paper links to the candidate the demo drilled into', () => {
  const findings = BEATS.find((b) => b.id === 'findings');
  // The loop the whole beat exists to close: the first proposal the top paper produced is
  // the candidate the drill-in beats walked through. The expected value is DERIVED from
  // DRILL_IN_ROW -- a literal id would keep passing while the two beats pointed at
  // different candidates, which is precisely the drift that would make the caption a lie.
  assert.equal(DRILL_IN_CAND, DRILL_IN_ROW.replace(/^tbody#/, ''));
  assert.equal(DRILL_IN_CAND, 'cand-vllm_v1_kv_offload-0009');
  assert.ok(DRILL_IN_ROW.endsWith(DRILL_IN_CAND), 'the two must name one candidate');

  const link = findings.asserts.find(
    (a) => a.type === 'attr' && a.name === 'data-cand',
  );
  assert.ok(link, 'the beat must assert which candidate the top paper produced');
  assert.equal(link.sel, TOP_FINDING_CAND_LINK);
  assert.equal(link.equals, DRILL_IN_CAND);
  // Not a hard-coded copy: the assertion's value IS the drill-in's, by identity.
  assert.equal(link.equals, DRILL_IN_ROW.replace(/^tbody#/, ''));

  // The link is pinned to the FIRST proposal of the TOP finding. Unpinned, `a[data-cand]`
  // would be satisfied by any proposal anywhere in the catalogue that happens to point at
  // the drill-in candidate, which proves nothing about this paper.
  assert.ok(TOP_FINDING_FIRST_PROP.startsWith(TOP_FINDING));
  assert.match(TOP_FINDING_FIRST_PROP, /details\.pitem >> nth=0$/);
  assert.equal(TOP_FINDING_CAND_LINK, `${TOP_FINDING_FIRST_PROP} >> a[data-cand]`);

  // data-cand is an attribute, and a viewer cannot read attributes. These two are what
  // make the tie legible in the recorded frame: the rank, and the symbol name.
  const rank = findings.asserts.find(
    (a) => a.type === 'textMatches' && a.sel === `${TOP_FINDING_FIRST_PROP} >> .prank`,
  );
  assert.ok(rank, 'the rendered rank needs an assertion');
  assert.equal(rank.pattern, '^#17$');
  assert.doesNotMatch('#170', new RegExp(rank.pattern));
  const sym = findings.asserts.find(
    (a) => a.type === 'textMatches' && a.sel === TOP_FINDING_CAND_LINK,
  );
  assert.ok(sym, 'the link text must name the symbol a viewer reads');
  assert.match('TieringOffloadingManager._initiate_promotion', new RegExp(sym.pattern));

  // And it is actually in frame, which no amount of DOM truth can witness.
  assert.ok(findings.asserts.some(
    (a) => a.type === 'inViewport' && a.sel === TOP_FINDING_CAND_LINK,
  ), 'the candidate link must be proved on screen');

  // Both captions say the loop out loud, so a viewer is told what the frame shows.
  for (const cut of ['gif', 'full']) {
    const caption = captionFor(findings, cut);
    assert.match(caption, /<b>#17<\/b>/, `${cut}: the rank the frame shows`);
    assert.match(caption, /candidate we/, `${cut}: the tie back to the drill-in`);
  }
  // The drill-in is the beat whose candidate this is, and it is earlier in the play order.
  const ids = BEATS.map((b) => b.id);
  assert.ok(ids.indexOf('drill-writeup') < ids.indexOf('findings'));
});

test('the proposals beat travels to the list in steps, not in one cut', () => {
  const props = BEATS.find((b) => b.id === 'drill-proposals');
  const step = props.actions.find((a) => a.type === 'scrollStepped');
  assert.ok(step, 'the proposals beat must scroll progressively');
  // No instant jump left in the beat: a scrollTo here would undo the whole point.
  assert.deepEqual(props.actions.filter((a) => a.type === 'scrollTo'), []);
  assert.deepEqual(props.actions.filter((a) => a.type === 'scrollAlign'), []);
  assert.equal(step.sel, `${DRILL_IN_ROW} .d-props`);
  assert.equal(step.block, 'start');
  // Enough hops, held long enough each, that the recording holds intermediate positions:
  // at 30fps a 90ms hop is ~2.7 frames, so every step is captured more than once.
  assert.ok(step.steps >= 5, 'a handful of hops is what makes the motion visible');
  assert.ok(step.stepMs >= 60, 'a hop shorter than a couple of frames would not record');
  assert.equal(step.steps, 10);
  assert.equal(step.stepMs, 90);

  // The travel ends with the list in frame, and that is asserted -- the three text and
  // child-count assertions read the DOM at any scroll position, so none of them can
  // witness where the scroll stopped.
  const cover = props.asserts.find((a) => a.type === 'viewportCoverage');
  assert.ok(cover, 'the beat must prove where the travel ended');
  assert.equal(cover.sel, step.sel, 'the assertion must watch what the beat scrolls');
  assert.equal(cover.minFraction, PROPOSALS_VIEWPORT_COVERAGE);
  // The list is ~4056px tall in an 810px frame, so inViewport could never pass; coverage
  // is the question worth asking, and the threshold has to reject the starting position.
  assert.deepEqual(props.asserts.filter((a) => a.type === 'inViewport'), []);
  assert.ok(PROPOSALS_VIEWPORT_COVERAGE > 0.30, 'the threshold must reject the write-up framing');
  assert.ok(PROPOSALS_VIEWPORT_COVERAGE <= 1);

  // It is the same candidate at both ends of the travel, which is what the motion says.
  const writeup = BEATS.find((b) => b.id === 'drill-writeup');
  assert.ok(writeup.actions.some((a) => a.sel.startsWith(DRILL_IN_ROW)));
  assert.ok(step.sel.startsWith(DRILL_IN_ROW));
});

/**
 * The guard on the framing defect that three beats shipped with, and that no assertion in
 * the storyboard could see: `block: 'start'` inside either of the page's tables lands the
 * aligned element underneath that table's own `position: sticky` header. Every text and
 * attribute assertion passes on a row the header is painted over, so the only witnesses
 * are the clearance itself and an occlusion check on what the clearance reveals.
 */
test('every start-aligned scroll inside a sticky-headered table carries a clearance', () => {
  const props = BEATS.find((b) => b.id === 'drill-proposals');
  const findings = BEATS.find((b) => b.id === 'findings');
  const lead = BEATS.find((b) => b.id === 'leaderboard-reveal');

  // Measured on the rendered page: the leaderboard's header box is 34px and the
  // catalogue's, whose label wraps to two lines, is 49px. Each clearance must clear its
  // own header, and neither may be so large it pushes the content it reveals out of frame.
  assert.ok(STICKY_CLEARANCE_LB > 34, 'the leaderboard header would still cover the row');
  assert.ok(STICKY_CLEARANCE_FND > 49, 'the catalogue header would still cover the row');
  for (const px of [STICKY_CLEARANCE_LB, STICKY_CLEARANCE_FND, CLOSE_CLEARANCE]) {
    assert.ok(px < GEOMETRY.height / 8, `a ${px}px clearance is framing, not clearance`);
  }

  const step = props.actions.find((a) => a.type === 'scrollStepped');
  assert.equal(step.marginTop, STICKY_CLEARANCE_LB);
  const frame = findings.actions.find((a) => a.type === 'scrollAlign');
  assert.equal(frame.marginTop, STICKY_CLEARANCE_FND);

  // The leaderboard reveal is the one start-aligned scroll that needs no clearance: it
  // aligns the section heading, which sits above the table rather than inside it.
  const reveal = lead.actions.find((a) => a.type === 'scrollAlign');
  assert.equal(reveal.marginTop, undefined);
  assert.equal(reveal.sel, LEADERBOARD_HEAD);
});

test('each clearance is witnessed by an occlusion check on what it reveals', () => {
  // A clearance is a number; what makes it testable at render time is asking the browser
  // what is actually painted at the revealed element's centre. inViewport cannot: the
  // findings title sat at y 10..33 under a 49px header and passed it.
  const expected = {
    'drill-proposals': [`${DRILL_IN_ROW} .d-props > h4`],
    findings: [
      `${TOP_FINDING} >> tr.frow .ftitle`,
      `${TOP_FINDING} >> tr.frow .tag`,
      `${TOP_FINDING} >> tr.frow .fhost`,
    ],
  };
  for (const [id, sels] of Object.entries(expected)) {
    const beat = BEATS.find((b) => b.id === id);
    assert.deepEqual(
      beat.asserts.filter((a) => a.type === 'unoccluded').map((a) => a.sel),
      sels,
      `${id} does not prove its clearance worked`,
    );
  }
  // Both cuts run both beats, so one clearance covers the gif and the mp4 alike.
  for (const id of Object.keys(expected)) {
    assert.deepEqual(BEATS.find((b) => b.id === id).cuts, ['gif', 'full']);
  }
});

test('the storyboard rejects a clearance or a bound that record.mjs would silently default', () => {
  // Both defaults are the failure: `marginTop ?? 0` is align-flush, the framing all this
  // exists to stop, and `box.y > undefined` is false, which passes from anywhere in frame.
  const align = { type: 'scrollAlign', sel: '#x', block: 'start' };
  const bad = [
    [{ ...align, marginTop: '48' }, /marginTop must be a non-negative number, got 48/],
    [{ ...align, marginTop: -8 }, /marginTop must be a non-negative number, got -8/],
    [{ ...align, marginTop: NaN }, /marginTop must be a non-negative number/],
  ];
  for (const [action, pattern] of bad) {
    const problems = withBeat({ actions: [action] });
    assert.ok(problems.some((m) => pattern.test(m)), `accepted ${JSON.stringify(action)}: ${problems}`);
  }
  // An omitted marginTop is legal -- most alignments want none -- so it must not be flagged.
  assert.deepEqual(withBeat({ actions: [align] }), []);

  assert.ok(
    withBeat({ asserts: [{ type: 'nearTop', sel: '#x' }] })
      .some((m) => /nearTop needs a non-negative maxY/.test(m)),
    'a nearTop with no bound asserts nothing and must be rejected',
  );
  assert.deepEqual(withBeat({ asserts: [{ type: 'nearTop', sel: '#x', maxY: 32 }] }), []);
});

test('the storyboard rejects a scroll alignment record.mjs would not understand', () => {
  // scrollAlign and scrollStepped share their alignments, and a typo would otherwise
  // reach the browser as a silent default rather than as a failure.
  assert.ok(ALIGN_BLOCKS instanceof Set);
  assert.deepEqual([...ALIGN_BLOCKS].sort(), ['center', 'start']);
  for (const beat of BEATS) {
    for (const a of beat.actions) {
      if (a.type === 'scrollAlign' || a.type === 'scrollStepped') {
        assert.ok(ALIGN_BLOCKS.has(a.block), `${beat.id}: ${a.type} has no known block`);
      }
    }
  }
  // Every aligning action in the storyboard declares its alignment explicitly.
  const aligning = BEATS.flatMap((b) => b.actions)
    .filter((a) => a.type === 'scrollAlign' || a.type === 'scrollStepped');
  // In play order: the leaderboard heading, the travel to the proposals, the tree, the
  // opened finding, the downstream artifacts. Only the tree is centred, and only because
  // it is taller than the frame.
  assert.equal(aligning.length, 5);
  assert.deepEqual(aligning.map((a) => a.block), ['start', 'start', 'center', 'start', 'start']);
});

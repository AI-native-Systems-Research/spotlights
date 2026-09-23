import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  BEATS, GEOMETRY, DRILL_IN_ROW, DRILL_IN_CAND, DRILL_IN_MODULE, DOWNSTREAM_SECTION,
  TREE_VIEWPORT_COVERAGE,
  FINDINGS_SECTION, TOP_FINDING, TOP_FINDING_FIRST_PROP, TOP_FINDING_CAND_LINK,
  LEADERBOARD_HEAD, PROPOSALS_VIEWPORT_COVERAGE, ALIGN_BLOCKS,
  STICKY_CLEARANCE_LB, STICKY_CLEARANCE_FND, SECTION_CLEARANCE, ALIGN_TOLERANCE_PX,
  FINDINGS_HEAD, LEADERBOARD_TOP_ROW,
  beatsForCut, dwellFor, captionFor, totalDuration, validateStoryboard, beatProblems,
} from './storyboard.mjs';

const ACTION_TYPES = new Set([
  // scrollCenter became scrollAlign, which takes the alignment as an option rather than
  // spelling one per action type; scrollStepped reaches the same place in visible hops.
  'scrollTo', 'scrollAlign', 'scrollStepped', 'click', 'select', 'hover', 'fill',
  // pickFromList is `select` made visible: the native dropdown is not captured by the
  // recorder, so the real control is expanded in-page and picked from.
  'pickFromList', 'ring', 'unring', 'pause',
  // expectText is an assertion shaped as an action, for a figure a beat shows and then
  // narrows away: `asserts` only ever witness the frame a beat ends on.
  'expectText',
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
  // `visible` on one of two mutually exclusive panes cannot witness a view switch.
  'hidden',
]);

test('the storyboard validates clean', () => {
  assert.deepEqual(validateStoryboard(), []);
});

test('geometry is the single recording geometry for both cuts', () => {
  assert.deepEqual(GEOMETRY, { width: 1440, height: 810 });
});

test('there are 14 beats with unique ids', () => {
  // 14, not 13: the catalogue is two beats now -- arriving at it, then asking what it is
  // for -- because one beat's assertions can only witness the frame it ends on.
  assert.equal(BEATS.length, 14);
  assert.equal(new Set(BEATS.map((b) => b.id)).size, 14);
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

test('the gif cut keeps exactly the nine marked beats, in order', () => {
  // Nine, not eight: the findings catalogue is in the short cut too, and it is two beats
  // -- the arrival and the payoff question. Both are in the gif: the arrival is what puts
  // the section's name on screen, without which the payoff beat opens on an unnamed table.
  assert.deepEqual(beatsForCut('gif').map((b) => b.id), [
    'cost-tiles', 'leaderboard-reveal', 'board', 'drill-writeup', 'drill-proposals',
    'radial-tree', 'findings', 'findings-paper', 'close',
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
  // 76.0, from 64.0: `board` +2.0 for opening the module list rather than setting it, the
  // 3.5s arrival beat split off the front of the catalogue, `close` +1.0 for the view
  // toggle back to the table, +2.5 correcting `radial-tree` (see below), and +3.0 for the
  // module filter the payoff beat grew -- a `pickFromList`, which opens a list and holds it
  // to be read rather than setting a value in one line.
  assert.equal(totalDuration('full'), 76.0);
  // 41.5, from 23.0: the same +2.0, +3.0, +1.0 and +3.0, less the 0.5 `findings-paper` hands
  // back in the short cut now that the arrival and its settle are the beat before it, plus
  // the +3.0, +1.0 and +6.0 of the three corrections.
  assert.equal(totalDuration('gif'), 41.5);
  // The arithmetic, spelled out, so a dwell that moves without its budget shows up as two
  // failures rather than one. It lives here rather than in its own test so the gif budget
  // is asserted in exactly one place -- the throwaway nortree variant re-keys this number,
  // and a second copy would widen what that patch breaks.
  const dwell = (id) => dwellFor(BEATS.find((b) => b.id === id), 'gif');
  const catalogue = dwell('findings') + dwell('findings-paper');
  assert.equal(catalogue, 11.5);
  assert.equal(Math.round((totalDuration('gif') - catalogue) * 10) / 10, 30.0);
  assert.equal(
    Math.round((totalDuration('gif') - catalogue - dwell('drill-proposals')) * 10) / 10,
    26.5,
  );
});

test('no dwell is smaller than the choreography it has to contain', () => {
  // A dwell is a deadline, not a duration: the recorder waits out whatever is left of it
  // and, when nothing is, logs an overrun and moves on. So a dwell below its own
  // choreography does not make the beat shorter -- it only stops predicting it, which is
  // how the gif budget came to read 28.5s for a recording that ran 37.2s.
  //
  // These are the measured costs, from `node record.mjs --cut <cut>` on the committed
  // storyboard: the choreography plus the 0.34s the recorder reserves for the caption's
  // fade-out, for the beats whose choreography is substantial enough to have crowded or
  // outrun a dwell. A beat whose real cost drifts past its dwell fails here rather than
  // silently at record time.
  const MEASURED_S = {
    gif: { board: 7.6, 'drill-writeup': 3.1, 'radial-tree': 7.1, 'findings-paper': 7.7 },
    full: { board: 7.9, 'radial-tree': 7.1, 'findings-paper': 7.7 },
  };
  for (const [cut, costs] of Object.entries(MEASURED_S)) {
    for (const [id, costS] of Object.entries(costs)) {
      const beat = BEATS.find((b) => b.id === id);
      assert.ok(
        dwellFor(beat, cut) >= costS,
        `${cut}/${id}: dwell ${dwellFor(beat, cut)}s cannot contain ${costS}s of choreography`,
      );
    }
  }
  // The two longest gif dwells, in order, and both for the same reason: they are the beats
  // that work a control the recorder cannot film collapsed. `findings-paper` now leads, on
  // three narrowing moves the last of which opens a list box; `board` is behind it on the
  // same open-list choreography plus undoing the drill-in's filters on camera. These are the
  // two places to look first if the gif has to give time back.
  const gifDwells = beatsForCut('gif').map((b) => [b.id, dwellFor(b, 'gif')])
    .sort((a, b) => b[1] - a[1]);
  assert.deepEqual(gifDwells.slice(0, 2).map((d) => d[0]), ['findings-paper', 'board']);
  assert.equal(dwellFor(BEATS.find((b) => b.id === 'radial-tree'), 'gif'), 7.5);
  // It costs the same in both cuts, because the choreography is identical in both.
  assert.equal(
    dwellFor(BEATS.find((b) => b.id === 'radial-tree'), 'gif'),
    dwellFor(BEATS.find((b) => b.id === 'radial-tree'), 'full'),
  );
});

test('the catalogue pays for the choreography it grew, across its two beats', () => {
  const arrival = BEATS.find((b) => b.id === 'findings');
  const payoff = BEATS.find((b) => b.id === 'findings-paper');
  // The payoff beat's 9.0 buys three narrowing moves, the click and the re-frame -- 7.4s of
  // the dwell-as-deadline budget before the caption's fade-out reserve, most of the growth
  // being the module filter opening its list. Its gif dwell is the one dwell in the
  // storyboard that is *lower* than its full one, because the arrival it used to do itself
  // is now a beat of its own and this caption is a line shorter.
  assert.equal(dwellFor(payoff, 'full'), 9.0);
  assert.equal(dwellFor(payoff, 'gif'), 8.5);
  // The arrival is a stepped travel plus a held frame, which is most of its 3.5s: ten 90ms
  // hops, a 200ms landing, a 500ms pause and the caption's own fade in and out.
  assert.equal(dwellFor(arrival, 'full'), 3.5);
  assert.equal(dwellFor(arrival, 'gif'), 3.0);
  // What the catalogue costs in total, in each cut. This is the number to look at first if
  // the gif has to give time back: the catalogue is now the demo's longest stretch, and the
  // module filter's open-list choreography is the most recent 3.0s of it.
  assert.equal(Math.round((dwellFor(arrival, 'full') + dwellFor(payoff, 'full')) * 10) / 10, 12.5);
  assert.equal(Math.round((dwellFor(arrival, 'gif') + dwellFor(payoff, 'gif')) * 10) / 10, 11.5);
});

/**
 * How many rendered characters fit on two lines of the caption card.
 *
 * The card is 1040px wide with 44px of horizontal padding at 26px in the report's sans,
 * whose average advance is about 12.5px -- roughly 80 characters a line. Two lines is the
 * ceiling because the card is 56px off the bottom: a third line reaches up into the frame's
 * content, which is how the fused catalogue caption came to lie across the downstream tiles.
 */
const TWO_CAPTION_LINES_CH = 160;

test('the catalogue is two beats in both cuts, arrival then payoff, before the close', () => {
  for (const id of ['findings', 'findings-paper']) {
    const beat = BEATS.find((b) => b.id === id);
    assert.ok(beat, `${id} must exist`);
    assert.deepEqual(beat.cuts, ['gif', 'full'], `${id} is in both cuts`);
  }
  const ids = BEATS.map((b) => b.id);
  // The sequence: this candidate, then the whole run as a tree, then where the evidence
  // came from, then which candidate it produced. Asserted on the shared array, which IS
  // the play order, so this pins both cuts at once.
  assert.equal(ids[ids.indexOf('drill-proposals') + 1], 'radial-tree');
  assert.deepEqual(ids.slice(-4), ['radial-tree', 'findings', 'findings-paper', 'close']);
  // The payoff beat must follow the arrival immediately: it has no scroll of its own and
  // works on the frame the arrival establishes, so anything between them breaks it.
  assert.equal(ids[ids.indexOf('findings') + 1], 'findings-paper');
  // The catalogue used to sit before the board, upstream of the drill-in. It must not any
  // more: the payoff beat points at the candidate the drill-in opened, which has to have
  // happened first.
  assert.ok(ids.indexOf('findings-paper') > ids.indexOf('board'));
  assert.ok(ids.indexOf('findings-paper') > ids.indexOf('drill-proposals'));
  for (const cut of ['gif', 'full']) {
    const order = beatsForCut(cut).map((b) => b.id);
    assert.equal(order[order.indexOf('radial-tree') + 1], 'findings');
    assert.equal(order[order.indexOf('findings') + 1], 'findings-paper');
    assert.equal(order[order.indexOf('findings-paper') + 1], 'close');
  }
});

test('the arrival beat frames the catalogue by its heading, and travels there in steps', () => {
  const arrival = BEATS.find((b) => b.id === 'findings');
  // A stepped travel, not a jump and not scrollIntoViewIfNeeded: the tree the previous
  // beat filled the frame with sits below the catalogue, so a cut reads as another page.
  assert.deepEqual(arrival.actions, [
    {
      type: 'scrollStepped',
      sel: FINDINGS_HEAD,
      block: 'start',
      marginTop: SECTION_CLEARANCE,
      steps: 10,
      stepMs: 90,
    },
    { type: 'pause', ms: 500 },
  ]);
  assert.equal(FINDINGS_HEAD, `${FINDINGS_SECTION} > h2`);

  // And it witnesses the frame it exists to establish. inViewport on the section would
  // not: the catalogue is 723px of a 810px frame, so it fits from a range of positions,
  // including ones with the heading above the top edge -- which is what the unsplit beat
  // silently recorded. These two say where the alignment landed and that nothing covers it.
  const anchored = arrival.asserts.find((a) => a.type === 'nearTop');
  assert.ok(anchored, 'the arrival must assert the frame is anchored on the heading');
  assert.equal(anchored.sel, FINDINGS_HEAD);
  assert.equal(anchored.maxY, SECTION_CLEARANCE + ALIGN_TOLERANCE_PX);
  assert.ok(
    arrival.asserts.some((a) => a.type === 'unoccluded' && a.sel === FINDINGS_HEAD),
    'the heading must be the element on top, not merely near the top',
  );
  // The controls the payoff beat reaches for, on screen before it starts.
  assert.ok(arrival.asserts.some((a) => a.type === 'inViewport' && a.sel === '#fq'));
  // Unfiltered here: the narrowing is the next beat's move, and the two counters together
  // are what witness it.
  const count = arrival.asserts.find((a) => a.type === 'textMatches' && a.sel === '#fcount');
  assert.equal(count.pattern, '^97 / 97 shown$');
});

test('the payoff beat narrows three times, and each move changes the table on camera', () => {
  const findings = BEATS.find((b) => b.id === 'findings-paper');
  // No scroll of its own: it opens on the frame the arrival beat established and asserted.
  assert.deepEqual(
    findings.actions.filter((a) => a.type.startsWith('scroll') && a.sel !== TOP_FINDING),
    [],
    'the arrival is the beat before this one; the only scroll here is the re-frame',
  );

  // The three moves, in the order that makes each one visible. This order is not a
  // preference: applied before the sort, the module filter would leave the sort nothing to
  // reorder, because the six kv_offload papers already sit in proposal order in the
  // document -- and a control that changes on camera while the table does not is exactly
  // the vacuous frame this storyboard keeps removing.
  const controls = findings.actions.filter(
    (a) => a.type === 'select' || a.type === 'pickFromList',
  );
  assert.deepEqual(controls, [
    { type: 'select', sel: '#fsrc', value: 'paper' },
    { type: 'select', sel: '#fsort', value: 'props:-1' },
    { type: 'pickFromList', sel: '#fmod', value: DRILL_IN_MODULE },
  ], 'source filter, then sort, then module filter');

  // The module filter is the last control the beat touches, so the frame it ends on is the
  // fully narrowed one, and it is filmed from an open list rather than set silently -- the
  // same reason the leaderboard's `#mod` is, since a collapsed <select> is a native widget
  // the recording cannot see.
  const types = findings.actions.map((a) => a.type);
  assert.equal(types.lastIndexOf('pickFromList'), findings.actions.indexOf(controls[2]));
  assert.ok(
    findings.actions.indexOf(controls[2]) > types.lastIndexOf('select'),
    'the module filter lands after both selects',
  );
  // It points at the same module the leaderboard was filtered to, through one constant:
  // the demo's claim is that the two filters are aimed at the same place.
  const board = BEATS.find((b) => b.id === 'board');
  assert.equal(
    board.actions.find((a) => a.sel === '#mod').value,
    controls[2].value,
    'the catalogue and the leaderboard must be narrowed to the same module',
  );

  assert.deepEqual(findings.actions[findings.actions.length - 1], { type: 'pause', ms: 600 });
  // It no longer types into the search box: `actions` is shared by both cuts and the
  // fill-pause-clear does not fit the gif's dwell.
  assert.deepEqual(findings.actions.filter((a) => a.type === 'fill'), []);
  // It must not touch the one catalogue control whose figures it does not quote.
  for (const a of findings.actions) {
    assert.notEqual(a.sel, '#fuse', 'the beat quotes no used/unused figure');
  }
});

test('the payoff caption\'s 27 is checked in the moment it is true, not at the end', () => {
  const findings = BEATS.find((b) => b.id === 'findings-paper');
  // The caption opens on 27 and the beat ends on 6, because the module filter narrows the
  // 27 papers further. A beat's `asserts` only ever see its last frame, so without a
  // checkpoint the 27 would be the one figure in the demo that nothing checks -- while
  // being on screen, under the caption that quotes it, for about a second.
  const checkpoints = findings.actions.filter((a) => a.type === 'expectText');
  assert.deepEqual(checkpoints, [
    { type: 'expectText', sel: '#fcount', pattern: '^27 / 97 shown$' },
  ]);
  assert.match('27 / 97 shown', new RegExp(checkpoints[0].pattern));
  assert.doesNotMatch('6 / 97 shown', new RegExp(checkpoints[0].pattern));
  assert.doesNotMatch('127 / 97 shown', new RegExp(checkpoints[0].pattern));

  // Position is the whole point of it being an action: it has to sit after the source
  // filter, which produces the 27, and before the module filter, which destroys it.
  const at = (pred) => findings.actions.findIndex(pred);
  const src = at((a) => a.sel === '#fsrc');
  const mod = at((a) => a.sel === '#fmod');
  const check = at((a) => a.type === 'expectText');
  assert.ok(src >= 0 && mod >= 0 && check > src && check < mod,
    `the checkpoint must sit between #fsrc (${src}) and #fmod (${mod}), not at ${check}`);
});

test('each catalogue caption carries exactly the figures its own beat proves', () => {
  const arrival = BEATS.find((b) => b.id === 'findings');
  const payoff = BEATS.find((b) => b.id === 'findings-paper');
  for (const cut of ['gif', 'full']) {
    // The arrival narrates the catalogue's two totals, which are the two figures visible
    // in the frame it establishes.
    const first = captionFor(arrival, cut);
    assert.match(first, /<b>97<\/b> findings/, `${cut}: the catalogue's own total`);
    assert.match(first, /<b>46<\/b> sites/, `${cut}: where the findings were published`);
    // ...and not the payoff figures, which nothing on screen supports yet.
    assert.doesNotMatch(first, /<b>27<\/b>/, `${cut}: the filter has not happened yet`);
    assert.doesNotMatch(first, /#17/, `${cut}: no row is open yet`);

    // The payoff beat narrates the chain it performs.
    const second = captionFor(payoff, cut);
    assert.match(second, /<b>27<\/b> from (?:type )?papers/, `${cut}: the filtered count`);
    assert.match(second, /<b>6<\/b> proposals|produced <b>6<\/b>/, `${cut}: the top row's payoff`);
    assert.match(second, /<b>#17<\/b>/, `${cut}: the candidate the chain lands on`);
  }
  // Splitting it is what got the caption back onto two lines: over a 1040px card at 26px
  // the fused sentence wrapped to three, far enough up the frame to lie across the tiles.
  // Measured on the rendered text, not the markup -- the <b> tags cost no width, so a
  // length check over the raw string would pass a sentence purely by being plainer.
  const rendered = (beat, cut) => captionFor(beat, cut).replace(/<[^>]+>/g, '');
  for (const beat of [arrival, payoff]) {
    assert.ok(rendered(beat, 'gif').length < rendered(beat, 'full').length);
    assert.ok(
      rendered(beat, 'full').length <= TWO_CAPTION_LINES_CH,
      `${beat.id}: the full caption is ${rendered(beat, 'full').length} rendered chars, ` +
        `over the ${TWO_CAPTION_LINES_CH} that fit two lines`,
    );
  }
  assert.match(captionFor(arrival, 'full'), /whether it paid off/);
});

test('each catalogue figure is witnessed through a scoped selector, in its own beat', () => {
  const arrival = BEATS.find((b) => b.id === 'findings');
  const payoff = BEATS.find((b) => b.id === 'findings-paper');
  const text = (beat, sel) => beat.asserts.find((a) => a.type === 'textMatches' && a.sel === sel);

  // Where the two filters leave the counter, anchored end to end. The caption's other
  // figure, the 27, is on screen a second earlier and is witnessed by the beat's mid-beat
  // `expectText` checkpoint instead -- `asserts` can only see the closing frame.
  const count = text(payoff, '#fcount');
  assert.ok(count, 'the filtered count needs an assertion on #fcount');
  assert.equal(count.pattern, '^6 / 97 shown$');
  assert.match('6 / 97 shown', new RegExp(count.pattern));
  assert.doesNotMatch('16 / 97 shown', new RegExp(count.pattern));
  assert.doesNotMatch('27 / 97 shown', new RegExp(count.pattern));
  assert.doesNotMatch('97 / 97 shown', new RegExp(count.pattern));
  // The intermediate reading, from the same counter, checked where it is true.
  assert.ok(
    payoff.actions.some((a) => a.type === 'expectText' && a.sel === '#fcount'
      && a.pattern === '^27 / 97 shown$'),
    "the caption's 27 needs a checkpoint on #fcount between the two filters",
  );
  // The same counter, unfiltered, in the beat before: across the three readings the whole
  // narrowing is witnessed rather than assumed from one of them.
  assert.equal(text(arrival, '#fcount').pattern, '^97 / 97 shown$');

  // The section's own 97, from the section's own heading, in the beat that frames it.
  const heading = text(arrival, FINDINGS_HEAD);
  assert.ok(heading, "the section's 97 needs an assertion on the section heading");
  assert.ok(
    arrival.asserts.some((a) => a.type === 'textMatches' && a.sel === FINDINGS_HEAD
      && /97 findings/.test(a.pattern)),
    'one of the heading assertions must carry the 97',
  );
  assert.ok(FINDINGS_SECTION.includes('#fq'), 'the section is addressed by the control it owns');

  // 46 sites, from the label the sort control states it under.
  const sites = text(arrival, '#fsort option[value="host:1"]');
  assert.ok(sites, 'the 46 sites needs an assertion on the sort option that states it');
  assert.match(sites.pattern, /46 sites/);

  // The 97, the 46 and the 27 are never taken from a document-wide substring check: the
  // page repeats all three, so such a check would pass with the catalogue gone.
  for (const beat of [arrival, payoff]) {
    assert.deepEqual(beat.asserts.filter((a) => a.type === 'domContains'), [], beat.id);
  }
});

test('the payoff beat proves the top row after the sort, not merely that some row matches', () => {
  const findings = BEATS.find((b) => b.id === 'findings-paper');
  // Pinned to one element: :visible skips the filtered-out tbodies the sort reorders
  // anyway, and nth=0 is what makes it the top row rather than any row.
  assert.ok(TOP_FINDING.includes(':visible'));
  assert.match(TOP_FINDING, /nth=0$/);

  const attrs = findings.asserts.filter((a) => a.type === 'attr' && a.sel === TOP_FINDING);
  assert.deepEqual(attrs, [
    // First, because it is what the beat's last move narrowed to, and because it is the
    // tie to the candidate the demo drilled into. The attribute is the filter's own key --
    // `#fmod` itself cannot be asserted on, since a <select>'s innerText is every option
    // concatenated and any pattern over it would pass.
    { type: 'attr', sel: TOP_FINDING, name: 'data-module', equals: DRILL_IN_MODULE },
    { type: 'attr', sel: TOP_FINDING, name: 'data-src', equals: 'paper' },
    { type: 'attr', sel: TOP_FINDING, name: 'data-props', equals: '6' },
  ], 'the top row must be proved a paper finding, in the module, with its proposal count');

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
      a.sel.startsWith(TOP_FINDING) || a.sel === '#fcount',
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
  // The click first, then the alignment. The two leaderboard views are different heights,
  // so aligning before the switch would compute the closing frame against a layout the
  // frame never has.
  assert.deepEqual(close.actions, [
    { type: 'click', sel: '#lbview' },
    { type: 'scrollAlign', sel: DOWNSTREAM_SECTION, block: 'start', marginTop: SECTION_CLEARANCE },
  ]);
  assert.deepEqual(close.actions.filter((a) => a.type === 'scrollTo'), []);
  assert.ok(
    DOWNSTREAM_SECTION.includes('Downstream artifacts'),
    'the close beat must scroll to the downstream artifacts section, not the leaderboard',
  );
  // It must no longer share the radial-tree beat's target, which is why it never moved.
  const tree = BEATS.find((b) => b.id === 'radial-tree');
  const treeScrolls = tree.actions.filter((a) => a.type === 'scrollTo').map((a) => a.sel);
  assert.ok(treeScrolls.includes('#lbview'));
  // The close does press #lbview -- that is how it gets back to the table -- but it must
  // not take its framing from the toggle, which is what left the shot where it was.
  for (const a of close.actions.filter((x) => x.type.startsWith('scroll'))) {
    assert.notEqual(a.sel, '#lbview');
  }

  // A domContains cannot witness scroll position; the beat needs an assertion that can.
  const inView = close.asserts.filter((a) => a.type === 'inViewport').map((a) => a.sel);
  assert.ok(inView.includes(DOWNSTREAM_SECTION), 'the beat must watch what it scrolls to');
  const scroll = close.actions.find((a) => a.type === 'scrollAlign');
  assert.ok(inView.includes(scroll.sel), 'the assertion must watch what the beat scrolls to');

  // inViewport alone is what let the stranded framing through: the section is 129px tall,
  // so it fitted from y 681..810 -- the bottom strip of a frame that was otherwise the
  // previous beat -- with a fraction of a pixel to spare. Only a bound on where its top
  // edge landed tells the two frames apart, and it has to be tight enough to reject the
  // bottom of the frame by a wide margin.
  const anchored = close.asserts.filter((a) => a.type === 'nearTop');
  assert.equal(anchored.length, 1);
  assert.equal(anchored[0].sel, DOWNSTREAM_SECTION);
  assert.equal(anchored[0].maxY, SECTION_CLEARANCE + ALIGN_TOLERANCE_PX);
  assert.ok(anchored[0].maxY < GEOMETRY.height / 8, 'a loose bound would re-admit the defect');
  // The figure guards stay.
  assert.deepEqual(
    close.asserts.filter((a) => a.type === 'domContains').map((a) => a.text),
    ['Awaiting measurement', 'no candidate has a measured speedup yet'],
  );
});

test('the close beat leaves the leaderboard as a table, not as the tree it came from', () => {
  const close = BEATS.find((b) => b.id === 'close');
  // The beat before this one fills the frame with the radial tree, so without the toggle
  // the demo's last shot of the leaderboard is a tree -- and the ranked table, which is
  // the artifact the whole run produces, is never the thing the viewer leaves on.
  assert.ok(BEATS.map((b) => b.id).indexOf('radial-tree') < BEATS.map((b) => b.id).indexOf('close'));
  assert.ok(
    BEATS.find((b) => b.id === 'radial-tree').asserts
      .some((a) => a.type === 'visible' && a.sel === '#lbtree'),
    'the beat this one undoes must be the one that showed the tree',
  );

  // visible on #lbtable cannot witness the switch by itself: the two panes are toggled
  // with the `hidden` attribute, and the assertion would read the same in either state if
  // the click silently failed to register -- the tree's pane is what has to be gone.
  assert.ok(close.asserts.some((a) => a.type === 'visible' && a.sel === '#lbtable'));
  assert.ok(
    close.asserts.some((a) => a.type === 'hidden' && a.sel === '#lbtree'),
    'the tree pane must be proved hidden, which is the half a visible check cannot see',
  );

  // And the control's own two readings, which is what makes the state legible in the
  // frame rather than only in the DOM: the button offers the trip back.
  assert.ok(close.asserts.some(
    (a) => a.type === 'attr' && a.sel === '#lbview' && a.name === 'aria-pressed' && a.equals === 'false',
  ));
  const label = close.asserts.find((a) => a.type === 'textMatches' && a.sel === '#lbview');
  assert.equal(label.pattern, '^radial tree$');
  assert.match('radial tree', new RegExp(label.pattern));
  assert.doesNotMatch('table view', new RegExp(label.pattern));

  // A table with no visible row would satisfy every check above, so the closing frame has
  // to prove a rank-ordered row is on screen and painted.
  assert.ok(close.asserts.some((a) => a.type === 'inViewport' && a.sel === LEADERBOARD_TOP_ROW));
  assert.ok(close.asserts.some((a) => a.type === 'unoccluded' && a.sel === LEADERBOARD_TOP_ROW));
  assert.ok(LEADERBOARD_TOP_ROW.includes(':visible'), 'the row must be one the filters left in');
  assert.match(LEADERBOARD_TOP_ROW, /nth=0 >> tr\.row$/, 'the top row, not any row');
});

test('the payoff beat opens the top paper finding rather than stopping at the sort', () => {
  const findings = BEATS.find((b) => b.id === 'findings-paper');
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

test('the payoff beat proves the top paper links to the candidate the demo drilled into', () => {
  const findings = BEATS.find((b) => b.id === 'findings-paper');
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
  const findings = BEATS.find((b) => b.id === 'findings-paper');
  const arrival = BEATS.find((b) => b.id === 'findings');
  const lead = BEATS.find((b) => b.id === 'leaderboard-reveal');

  // Measured on the rendered page: the leaderboard's header box is 34px and the
  // catalogue's, whose label wraps to two lines, is 49px. Each clearance must clear its
  // own header, and neither may be so large it pushes the content it reveals out of frame.
  assert.ok(STICKY_CLEARANCE_LB > 34, 'the leaderboard header would still cover the row');
  assert.ok(STICKY_CLEARANCE_FND > 49, 'the catalogue header would still cover the row');
  for (const px of [STICKY_CLEARANCE_LB, STICKY_CLEARANCE_FND, SECTION_CLEARANCE]) {
    assert.ok(px < GEOMETRY.height / 8, `a ${px}px clearance is framing, not clearance`);
  }

  const step = props.actions.find((a) => a.type === 'scrollStepped');
  assert.equal(step.marginTop, STICKY_CLEARANCE_LB);
  const frame = findings.actions.find((a) => a.type === 'scrollAlign');
  assert.equal(frame.marginTop, STICKY_CLEARANCE_FND);

  // The catalogue's arrival and the close both align a section heading rather than a row
  // inside a table, so they clear the page's own sticky topbar and not a table header --
  // the smaller of the three numbers, and the same one for both.
  assert.equal(arrival.actions.find((a) => a.type === 'scrollStepped').marginTop, SECTION_CLEARANCE);
  assert.equal(
    BEATS.find((b) => b.id === 'close').actions.find((a) => a.type === 'scrollAlign').marginTop,
    SECTION_CLEARANCE,
  );
  assert.ok(SECTION_CLEARANCE < STICKY_CLEARANCE_LB, 'a section heading clears less than a table row');

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
    // The arrival's clearance reveals exactly one thing: the heading it aligns on.
    findings: [FINDINGS_HEAD],
    'findings-paper': [
      `${TOP_FINDING} >> tr.frow .ftitle`,
      `${TOP_FINDING} >> tr.frow .tag`,
      `${TOP_FINDING} >> tr.frow .fhost`,
    ],
    // The close aligns the downstream section, and what its clearance has to reveal below
    // that is the leaderboard's own top row -- back in table view, which is the point.
    close: [LEADERBOARD_TOP_ROW],
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
  // catalogue's heading, the opened finding, the downstream artifacts. Only the tree is
  // centred, and only because it is taller than the frame.
  assert.equal(aligning.length, 6);
  assert.deepEqual(
    aligning.map((a) => a.block),
    ['start', 'start', 'center', 'start', 'start', 'start'],
  );
});

test('the board beat picks the module from an open list, because a select cannot be filmed', () => {
  const board = BEATS.find((b) => b.id === 'board');
  // Chrome draws a <select>'s dropdown as a native OS widget, outside the page's
  // compositing surface. Playwright's video and screenshots capture the page, so a plain
  // `select` action records as a value that changes with no list ever appearing and no
  // visible cause -- verified by screenshotting a real click on #mod. pickFromList expands
  // the same control in-page (size > 1) so the options are painted pixels.
  assert.ok(
    board.actions.some((a) => a.type === 'pickFromList' && a.sel === '#mod'
      && a.value === 'vllm/v1/kv_offload'),
    'the module choice must be made from a list the camera can see',
  );
  assert.deepEqual(
    board.actions.filter((a) => a.type === 'select' && a.sel === '#mod'),
    [],
    'the beat that demonstrates the choice must not make it invisibly',
  );

  // It happens after the hot-modules toggle, so the list the viewer reads is the hot one:
  // the option labels carry each module's share, which is what the toggle just added.
  const types = board.actions.map((a) => a.type);
  assert.ok(types.indexOf('pickFromList') > types.indexOf('click'), 'pick after the toggle');
  assert.equal(board.actions[types.indexOf('click')].sel, '#hotbtn');

  // The counter is what witnesses WHICH module was picked, so it is anchored to the one
  // number only this module produces. Hot mode on its own leaves 86 of the 173, so the
  // loose `^\d+ / 173 shown$` this replaced passed on the toggle alone -- with the pick
  // silently absent, which is exactly the defect the native dropdown was hiding.
  const count = board.asserts.find((a) => a.type === 'textMatches' && a.sel === '#count');
  assert.equal(count.pattern, '^19 / 173 shown$');
  assert.doesNotMatch('86 / 173 shown', new RegExp(count.pattern));
  assert.doesNotMatch('119 / 173 shown', new RegExp(count.pattern));
  // Not asserted on #mod itself: a <select>'s innerText is every option's label
  // concatenated, so any pattern over it matches whatever the selection is.
  assert.deepEqual(board.asserts.filter((a) => a.sel === '#mod'), []);

  // Elsewhere, a plain `select` stays correct: the radial-tree beat clears this same
  // filter, and a reset is bookkeeping between shots rather than a move being shown.
  assert.ok(
    BEATS.find((b) => b.id === 'radial-tree').actions
      .some((a) => a.type === 'select' && a.sel === '#mod' && a.value === ''),
  );
});

test('the storyboard rejects a pickFromList with nothing to pick', () => {
  // record.mjs passes `value` to selectOption and also builds the option selector from it,
  // so an omitted or empty value is not a no-op: it hunts for `option[value=""]`, which on
  // #mod is the "all modules" entry -- a pick that silently means the opposite.
  for (const value of [undefined, '', 42, null]) {
    const problems = withBeat({ actions: [{ type: 'pickFromList', sel: '#mod', value }] });
    assert.ok(
      problems.some((m) => /pickFromList needs a non-empty value/.test(m)),
      `accepted value ${JSON.stringify(value)}: ${problems}`,
    );
  }
  assert.deepEqual(
    withBeat({ actions: [{ type: 'pickFromList', sel: '#mod', value: 'vllm/v1/kv_offload' }] }),
    [],
  );
});

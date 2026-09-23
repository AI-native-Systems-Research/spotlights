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
  // caption is a marker, not a move: it says where in the choreography the beat's caption
  // comes up, for the beats whose caption states a fact the choreography has yet to produce.
  'caption',
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

test('there are 15 beats with unique ids', () => {
  // 15, not 13: the catalogue is three beats -- arriving at it, narrowing to papers, then
  // narrowing to the module and opening the top one -- because one beat's assertions can
  // only witness the frame it ends on, and the catalogue passes through three states whose
  // figures its captions quote.
  assert.equal(BEATS.length, 15);
  assert.equal(new Set(BEATS.map((b) => b.id)).size, 15);
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

test('the gif cut keeps exactly the ten marked beats, in order', () => {
  // Ten, not eight: the findings catalogue is in the short cut too, and it is three beats
  // -- the arrival, the narrowing to papers, and the narrowing to the module that opens the
  // top paper. All three are in the gif: the arrival is what puts the section's name on
  // screen, without which the rest opens on an unnamed table, and the two filter beats each
  // end on a figure their own caption quotes.
  assert.deepEqual(beatsForCut('gif').map((b) => b.id), [
    'cost-tiles', 'leaderboard-reveal', 'board', 'drill-writeup', 'drill-proposals',
    'radial-tree', 'findings', 'findings-paper', 'findings-module', 'close',
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
  // 79.0, from 64.0: `board` +2.0 for opening the module list rather than setting it, the
  // 3.5s arrival beat split off the front of the catalogue, `close` +1.0 for the view
  // toggle back to the table, +2.5 correcting `radial-tree` (see below), +3.0 for the module
  // filter -- a `pickFromList`, which opens a list and holds it to be read rather than
  // setting a value in one line -- and +3.0 for splitting the filter beat in two, of which
  // only ~1.2s is new choreography and the rest is the second caption's own transitions and
  // reading time, which is what the split was for.
  assert.equal(totalDuration('full'), 82.5);
  // 44.0, from 23.0: the same +2.0, +3.0, +1.0, +3.0 and +3.0, less the 0.5 the papers beat
  // hands back in the short cut now that the arrival and its settle are before it, plus the
  // +3.0, +1.0 and +6.0 of the three corrections.
  assert.equal(totalDuration('gif'), 47.5);
  // The arithmetic, spelled out, so a dwell that moves without its budget shows up as two
  // failures rather than one. It lives here rather than in its own test so the gif budget
  // is asserted in exactly one place -- the throwaway nortree variant re-keys this number,
  // and a second copy would widen what that patch breaks.
  const dwell = (id) => dwellFor(BEATS.find((b) => b.id === id), 'gif');
  const catalogue = dwell('findings') + dwell('findings-paper') + dwell('findings-module');
  assert.equal(catalogue, 17.5);
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
    gif: {
      board: 7.6, 'drill-writeup': 3.1, 'radial-tree': 7.1,
      findings: 2.8, 'findings-paper': 5.0, 'findings-module': 6.2,
    },
    full: {
      board: 7.9, 'radial-tree': 7.1,
      findings: 2.8, 'findings-paper': 5.0, 'findings-module': 6.2,
    },
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
  // The three longest gif dwells, in order. These are the places to look first if the gif
  // has to give time back, and each is long for a stated reason rather than by taste:
  // `board` opens the module list and then undoes the drill-in's filters on camera,
  // `radial-tree` spends nearly all of its time resetting the board to all 173 as a tree,
  // and `findings-module` opens the catalogue's own module list before the click and the
  // re-frame. Two of the three are the cost of filming a `<select>` honestly.
  const gifDwells = beatsForCut('gif').map((b) => [b.id, dwellFor(b, 'gif')])
    .sort((a, b) => b[1] - a[1]);
  assert.deepEqual(gifDwells.slice(0, 3).map((d) => d[0]),
    ['board', 'radial-tree', 'findings-module']);
  assert.equal(dwellFor(BEATS.find((b) => b.id === 'radial-tree'), 'gif'), 7.5);
  // It costs the same in both cuts, because the choreography is identical in both.
  assert.equal(
    dwellFor(BEATS.find((b) => b.id === 'radial-tree'), 'gif'),
    dwellFor(BEATS.find((b) => b.id === 'radial-tree'), 'full'),
  );
});

test('the catalogue pays for the choreography it grew, across its three beats', () => {
  const arrival = BEATS.find((b) => b.id === 'findings');
  const papers = BEATS.find((b) => b.id === 'findings-paper');
  const module = BEATS.find((b) => b.id === 'findings-module');
  // The arrival is a stepped travel plus a held frame: ten 90ms hops, a 200ms landing, a
  // 500ms pause and the caption's own fade in and out. Its dwell carries the travel twice
  // over, in effect, because the caption is cued after it -- the ~1.95s of scrolling is
  // beat time that is not caption time.
  assert.equal(dwellFor(arrival, 'full'), 5.5);
  assert.equal(dwellFor(arrival, 'gif'), 5.0);
  // The papers beat opens the source list and then sorts: 4.63s of measured choreography,
  // of which the ~1.9s list-opening happens before its caption is cued.
  assert.equal(dwellFor(papers, 'full'), 6.0);
  assert.equal(dwellFor(papers, 'gif'), 5.5);
  // The module beat carries the expensive half -- the open list, the click, the re-frame --
  // at 5.82s measured, and is the longer of the two for that reason rather than by taste.
  assert.equal(dwellFor(module, 'full'), 7.5);
  assert.equal(dwellFor(module, 'gif'), 7.0);
  assert.ok(dwellFor(module, 'gif') > dwellFor(papers, 'gif'));
  // What the catalogue costs in total, in each cut. This is the number to look at first if
  // the gif has to give time back: the catalogue is the demo's longest stretch, and the two
  // module lists -- this one and the board's -- are the most expensive thing in it.
  const total = (cut) => Math.round(
    (dwellFor(arrival, cut) + dwellFor(papers, cut) + dwellFor(module, cut)) * 10,
  ) / 10;
  assert.equal(total('full'), 19.0);
  assert.equal(total('gif'), 17.5);
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

test('the catalogue is three beats in both cuts, in narrowing order, before the close', () => {
  for (const id of ['findings', 'findings-paper', 'findings-module']) {
    const beat = BEATS.find((b) => b.id === id);
    assert.ok(beat, `${id} must exist`);
    assert.deepEqual(beat.cuts, ['gif', 'full'], `${id} is in both cuts`);
  }
  const ids = BEATS.map((b) => b.id);
  // The sequence: this candidate, then the whole run as a tree, then where the evidence
  // came from, then which of it paid off, then which candidate it produced. Asserted on the
  // shared array, which IS the play order, so this pins both cuts at once.
  assert.equal(ids[ids.indexOf('drill-proposals') + 1], 'radial-tree');
  assert.deepEqual(ids.slice(-5),
    ['radial-tree', 'findings', 'findings-paper', 'findings-module', 'close']);
  // Each of the three must follow the one before it immediately: neither filter beat has a
  // scroll of its own, so both work on the frame the arrival establishes and anything
  // between them breaks that. The order is also the narrowing order -- 97, then 27, then 6
  // -- so a swap would put a caption over a table that does not match it.
  assert.equal(ids[ids.indexOf('findings') + 1], 'findings-paper');
  assert.equal(ids[ids.indexOf('findings-paper') + 1], 'findings-module');
  // The catalogue used to sit before the board, upstream of the drill-in. It must not any
  // more: the module beat points at the candidate the drill-in opened, which has to have
  // happened first.
  assert.ok(ids.indexOf('findings-paper') > ids.indexOf('board'));
  assert.ok(ids.indexOf('findings-module') > ids.indexOf('drill-proposals'));
  for (const cut of ['gif', 'full']) {
    const order = beatsForCut(cut).map((b) => b.id);
    assert.equal(order[order.indexOf('radial-tree') + 1], 'findings');
    assert.equal(order[order.indexOf('findings') + 1], 'findings-paper');
    assert.equal(order[order.indexOf('findings-paper') + 1], 'findings-module');
    assert.equal(order[order.indexOf('findings-module') + 1], 'close');
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
    // The caption comes up after the travel, not at the top of the beat: it names the
    // section, and for the ~1.95s the scroll is running the section is not on screen yet.
    { type: 'caption' },
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

test('the catalogue narrows three times, and each move changes the table on camera', () => {
  const papers = BEATS.find((b) => b.id === 'findings-paper');
  const module = BEATS.find((b) => b.id === 'findings-module');
  // Neither has a scroll of its own except the re-frame that follows the click: both open
  // on the frame the arrival beat established and asserted.
  for (const beat of [papers, module]) {
    assert.deepEqual(
      beat.actions.filter((a) => a.type.startsWith('scroll') && a.sel !== TOP_FINDING),
      [],
      `${beat.id}: the arrival established the frame; the only scroll is the re-frame`,
    );
  }

  // The three moves, in the order that makes each one visible, read across the two beats.
  // This order is not a preference: applied before the sort, the module filter would leave
  // the sort nothing to reorder, because the six kv_offload papers already sit in proposal
  // order in the document -- and a control that changes on camera while the table does not
  // is exactly the vacuous frame this storyboard keeps removing.
  const controlsOf = (beat) => beat.actions.filter(
    (a) => a.type === 'select' || a.type === 'pickFromList',
  );
  assert.deepEqual(controlsOf(papers), [
    { type: 'pickFromList', sel: '#fsrc', value: 'paper' },
    { type: 'select', sel: '#fsort', value: 'props:-1' },
  ], 'the papers beat filters by source, then sorts');
  assert.deepEqual(controlsOf(module), [
    { type: 'pickFromList', sel: '#fmod', value: DRILL_IN_MODULE },
  ], 'the module beat does exactly one narrowing move');

  // Both filters are filmed from an open list rather than set silently -- a collapsed
  // <select> is a native widget the recording cannot see, so a plain `select` would make
  // the filter the one move in the catalogue the viewer never watches happen. The sort is
  // the exception: it needs no list, because the table reordering underneath it is itself
  // the evidence that something was chosen.
  assert.deepEqual(
    [...controlsOf(papers), ...controlsOf(module)].filter((a) => a.type === 'select'),
    [{ type: 'select', sel: '#fsort', value: 'props:-1' }],
    'every filter is picked from an open list; only the sort is set silently',
  );

  // Each beat cues its caption rather than taking the default of raising it first, and
  // cues it after the move that makes the caption true: the papers beat's "27 from type
  // papers" after the source filter, the module beat's after the module filter.
  for (const [beat, sel] of [[papers, '#fsrc'], [module, '#fmod']]) {
    const at = (pred) => beat.actions.findIndex(pred);
    const cue = at((a) => a.type === 'caption');
    assert.ok(cue >= 0, `${beat.id}: its caption states a filtered figure, so it must be cued`);
    assert.ok(
      cue > at((a) => a.sel === sel),
      `${beat.id}: the caption must come up after ${sel}, not before it`,
    );
  }
  // And the module beat cues it before the click, so the viewer reads the claim and then
  // watches it checked rather than the other way round.
  assert.ok(
    module.actions.findIndex((a) => a.type === 'caption')
      < module.actions.findIndex((a) => a.type === 'click'),
  );

  // It points at the same module the leaderboard was filtered to, through one constant:
  // the demo's claim is that the two filters are aimed at the same place.
  const board = BEATS.find((b) => b.id === 'board');
  assert.equal(
    board.actions.find((a) => a.sel === '#mod').value,
    controlsOf(module)[0].value,
    'the catalogue and the leaderboard must be narrowed to the same module',
  );

  // The papers beat ends on its own last control, so the frame it leaves is the sorted 27.
  assert.equal(papers.actions[papers.actions.length - 1].sel, '#fsort');
  assert.deepEqual(module.actions[module.actions.length - 1], { type: 'pause', ms: 600 });
  // Neither types into the search box: `actions` is shared by both cuts and the
  // fill-pause-clear does not fit the gif's dwell.
  // Nor touches the one catalogue control whose figures no caption quotes.
  for (const beat of [papers, module]) {
    assert.deepEqual(beat.actions.filter((a) => a.type === 'fill'), [], beat.id);
    for (const a of beat.actions) {
      assert.notEqual(a.sel, '#fuse', `${beat.id}: no caption quotes a used/unused figure`);
    }
  }
});

test('each filter beat ends on the figure its own caption names', () => {
  // This is what the split bought. As one beat, the choreography passed through 27 / 97 and
  // then left it for 6 / 97 under a single caption naming both -- so the 27 was true for
  // about a second of the beat that quoted it, and a beat's `asserts`, which only ever
  // witness the frame it ends on, could not reach it. Split, each beat's closing frame is
  // the one its caption describes, and a plain assertion witnesses it.
  const countOf = (id) => BEATS.find((b) => b.id === id)
    .asserts.find((a) => a.type === 'textMatches' && a.sel === '#fcount');
  const readings = ['findings', 'findings-paper', 'findings-module'].map(countOf);
  assert.deepEqual(readings.map((a) => a && a.pattern), [
    '^97 / 97 shown$', '^27 / 97 shown$', '^6 / 97 shown$',
  ], 'the three beats must witness the three states of the counter, in narrowing order');
  // Anchored end to end, so a wider filter cannot satisfy a narrower beat's reading.
  for (const [pattern, wrong] of [
    ['^97 / 97 shown$', '197 / 97 shown'],
    ['^27 / 97 shown$', '127 / 97 shown'],
    ['^6 / 97 shown$', '16 / 97 shown'],
  ]) {
    assert.doesNotMatch(wrong, new RegExp(pattern));
  }
  // And each caption names the figure its own beat ends on, rather than one of the others.
  for (const cut of ['gif', 'full']) {
    assert.match(captionFor(BEATS.find((b) => b.id === 'findings-paper'), cut), /<b>27<\/b>/);
    assert.doesNotMatch(
      captionFor(BEATS.find((b) => b.id === 'findings-paper'), cut), /#17/,
      `${cut}: the papers beat has not opened anything yet`,
    );
  }
});

test('each catalogue caption carries exactly the figures its own beat proves', () => {
  const arrival = BEATS.find((b) => b.id === 'findings');
  const papers = BEATS.find((b) => b.id === 'findings-paper');
  const module = BEATS.find((b) => b.id === 'findings-module');
  for (const cut of ['gif', 'full']) {
    // The arrival narrates the catalogue's two totals, which are the two figures visible
    // in the frame it establishes.
    const first = captionFor(arrival, cut);
    assert.match(first, /<b>97<\/b> findings/, `${cut}: the catalogue's own total`);
    assert.match(first, /<b>46<\/b> sites/, `${cut}: where the findings were published`);
    // ...and not the figures the two filter beats go on to produce, which nothing on
    // screen supports yet.
    assert.doesNotMatch(first, /<b>27<\/b>/, `${cut}: the filter has not happened yet`);
    assert.doesNotMatch(first, /#17/, `${cut}: no row is open yet`);

    // The papers beat names its own count and nothing further along the chain. This is the
    // split's whole point: the figure it quotes is the one its closing frame shows.
    const second = captionFor(papers, cut);
    assert.match(second, /<b>27<\/b> from (?:type )?papers/, `${cut}: the filtered count`);
    assert.doesNotMatch(second, /<b>#17<\/b>/, `${cut}: nothing is open in this beat`);
    assert.doesNotMatch(second, /proposals/, `${cut}: the payoff is the next beat's claim`);

    // The module beat names what opening the top paper reveals -- and not the 27, which by
    // the time its caption is up has been narrowed away to 6.
    const third = captionFor(module, cut);
    assert.match(third, /<b>6<\/b> proposals|produced <b>6<\/b>/, `${cut}: the top row's payoff`);
    assert.match(third, /<b>#17<\/b>/, `${cut}: the candidate the chain lands on`);
    assert.doesNotMatch(third, /<b>27<\/b>/, `${cut}: the 27 is no longer on screen`);
  }
  // Every catalogue caption fits two lines. Over a 1040px card at 26px the fused sentence
  // this pair was split out of wrapped to three, far enough up the frame to lie across the
  // downstream tiles. Measured on the rendered text, not the markup -- the <b> tags cost no
  // width, so a length check over the raw string would pass a sentence purely by being
  // plainer.
  const rendered = (beat, cut) => captionFor(beat, cut).replace(/<[^>]+>/g, '');
  for (const beat of [arrival, papers, module]) {
    assert.ok(
      rendered(beat, 'gif').length < rendered(beat, 'full').length,
      `${beat.id}: the gif caption must be the shorter of the two`,
    );
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
  const papers = BEATS.find((b) => b.id === 'findings-paper');
  const module = BEATS.find((b) => b.id === 'findings-module');
  const text = (beat, sel) => beat.asserts.find((a) => a.type === 'textMatches' && a.sel === sel);

  // The counter's three readings live one per beat, each in the beat whose closing frame
  // shows it. `each filter beat ends on the figure its own caption names` pins the
  // sequence; this one pins that each reading is scoped to the catalogue's own counter
  // rather than read off the page at large.
  for (const beat of [arrival, papers, module]) {
    assert.ok(text(beat, '#fcount'), `${beat.id}: its count needs an assertion on #fcount`);
  }

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

  // None of the catalogue's figures is taken from a document-wide substring check: the
  // page repeats all of them, so such a check would pass with the catalogue gone.
  for (const beat of [arrival, papers, module]) {
    assert.deepEqual(beat.asserts.filter((a) => a.type === 'domContains'), [], beat.id);
  }
});

test('both filter beats prove the top row they leave, not merely that some row matches', () => {
  const papers = BEATS.find((b) => b.id === 'findings-paper');
  const module = BEATS.find((b) => b.id === 'findings-module');
  // Pinned to one element: :visible skips the filtered-out tbodies the sort reorders
  // anyway, and nth=0 is what makes it the top row rather than any row.
  assert.ok(TOP_FINDING.includes(':visible'));
  assert.match(TOP_FINDING, /nth=0$/);

  const attrs = (beat) => beat.asserts.filter((a) => a.type === 'attr' && a.sel === TOP_FINDING);
  // After the sort: a paper with six proposals at the top of the 27.
  assert.deepEqual(attrs(papers), [
    { type: 'attr', sel: TOP_FINDING, name: 'data-src', equals: 'paper' },
    { type: 'attr', sel: TOP_FINDING, name: 'data-props', equals: '6' },
  ], 'the sorted top row must be proved a paper finding with six proposals');
  // After the module filter: the same three facts again, plus the module itself. Repeated
  // rather than inherited, and not out of caution -- the module filter re-picks `nth=0`
  // from a different set of rows, so "the top row" in the two beats is two claims about two
  // different tables, and only the second one is the row this beat opens.
  assert.deepEqual(attrs(module), [
    // First, because it is what this beat's move narrowed to, and because it is the tie to
    // the candidate the demo drilled into. The attribute is the filter's own key -- `#fmod`
    // itself cannot be asserted on, since a <select>'s innerText is every option
    // concatenated and any pattern over it would pass.
    { type: 'attr', sel: TOP_FINDING, name: 'data-module', equals: DRILL_IN_MODULE },
    { type: 'attr', sel: TOP_FINDING, name: 'data-src', equals: 'paper' },
    { type: 'attr', sel: TOP_FINDING, name: 'data-props', equals: '6' },
  ], 'the narrowed top row must be proved a paper finding, in the module, with its count');

  // And in both beats, the two cells a viewer actually reads off the frame -- because the
  // attributes above are invisible, and what the caption claims is what is rendered.
  for (const beat of [papers, module]) {
    const badge = beat.asserts.find(
      (a) => a.type === 'textMatches' && a.sel === `${TOP_FINDING} >> tr.frow .tag`,
    );
    assert.ok(badge, `${beat.id}: the rendered source badge needs an assertion`);
    assert.equal(badge.pattern, '^paper$');
    const props = beat.asserts.find(
      (a) => a.type === 'textMatches' && a.sel === `${TOP_FINDING} >> tr.frow td.r >> nth=1`,
    );
    assert.ok(props, `${beat.id}: the rendered Proposals cell needs an assertion`);
    assert.equal(props.pattern, '^6$');
    // Anchored, so a cell reading 16 or 61 cannot satisfy the caption's 6.
    assert.doesNotMatch('16', new RegExp(props.pattern));
    assert.doesNotMatch('61', new RegExp(props.pattern));

    // Every assertion in the beat is scoped to the catalogue or to its top row.
    for (const a of beat.asserts) {
      assert.ok(
        a.sel.startsWith(TOP_FINDING) || a.sel === '#fcount',
        `${beat.id}: unscoped assertion selector: ${a.sel}`,
      );
    }
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

test('the module beat opens the top paper finding rather than stopping at the filter', () => {
  const findings = BEATS.find((b) => b.id === 'findings-module');
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

test('the module beat proves the top paper links to the candidate the demo drilled into', () => {
  const findings = BEATS.find((b) => b.id === 'findings-module');
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
  const findings = BEATS.find((b) => b.id === 'findings-module');
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
    'findings-module': [
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

test('the storyboard rejects a caption cue that cannot do what it says', () => {
  // A cue in a beat with no caption marks a moment for a sentence that never appears --
  // it does not fail at record time, it simply does nothing, which is how a beat could end
  // up silently uncaptioned after a caption was deleted.
  // Asserted as a member rather than as the whole list: a beat with no caption is already
  // a problem on its own account, and this rule is the second, independent one.
  assert.ok(
    withBeat({ caption: {}, actions: [{ type: 'caption' }] })
      .includes('probe: a caption cue in a beat with no caption'),
  );
  // Two cues do not fail at record time either: the recorder would raise the caption
  // twice, resetting the clock that measures its on-screen time, so the beat would report
  // a legible caption while having shown it for less than it claims.
  assert.deepEqual(
    withBeat({ actions: [{ type: 'caption' }, { type: 'caption' }] }),
    ['probe: 2 caption cues, want at most 1'],
  );
  // One cue in a captioned beat is the whole point, and passes.
  assert.deepEqual(withBeat({ actions: [{ type: 'caption' }] }), []);
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

/**
 * The demo walkthrough as data. No Playwright import: this module is pure so it
 * unit-tests without a browser, and `record.mjs` is the only thing that knows
 * how to execute an action or check an assertion.
 *
 * Every figure quoted in a caption is also listed in that beat's `asserts`, so a
 * number that drifts out of the report fails the render instead of shipping.
 */

export const GEOMETRY = { width: 1440, height: 810 };
export const PAGE_RELATIVE_PATH = 'examples/vllm_subset/experiment.html';

/** Rank 17, TieringOffloadingManager._initiate_promotion, 11 proposals, 5 paper-cited. */
export const DRILL_IN_ROW = 'tbody#cand-vllm_v1_kv_offload-0009';

/**
 * The downstream-artifacts section: the three tiles the closing caption is about.
 * The section carries no id, so it is addressed by its heading. Playwright's text
 * engine matches DOM text, which `text-transform: uppercase` on `section.sec > h2`
 * does not touch, so the source spelling is the right thing to match on.
 */
export const DOWNSTREAM_SECTION = 'section.sec:has(h2:text-is("Downstream artifacts"))';

/**
 * The share of the viewport's height the radial tree must fill in the tree beat.
 *
 * Measured, not guessed: the plot renders 1186px square in the 1440x810 viewport, so
 * once it is centred it covers the full 810px (coverage 1.00), while the toggle-only
 * scroll the beat used to end on leaves just 307px of it in frame (coverage 0.38).
 * 0.90 sits well clear of both -- it fails the old framing and holds with slack for
 * whatever the layout does to the plot's size.
 */
export const TREE_VIEWPORT_COVERAGE = 0.9;

const CUTS = new Set(['gif', 'full']);
const BOTH = ['gif', 'full'];
const FULL_ONLY = ['full'];

export const BEATS = [
  {
    id: 'the-ask',
    cuts: FULL_ONLY,
    dwell: { full: 4.0 },
    caption: { full: 'the objective: <b>reduce median TTFT and TPOT</b>' },
    actions: [
      { type: 'scrollTo', sel: '.topbar' },
      { type: 'ring', sel: '.topbar .obj' },
      { type: 'pause', ms: 600 },
    ],
    asserts: [
      { type: 'textMatches', sel: '.topbar .obj', pattern: 'median TTFT' },
      { type: 'domContains', text: 'TPOT' },
    ],
  },
  {
    id: 'cost-tiles',
    cuts: BOTH,
    dwell: { full: 4.5, gif: 2.5 },
    caption: {
      full: '<b>173</b> candidates · 60 high impact · <b>544</b> proposals · <b>$35.75</b> · 266.42M tokens, <b>85% cached</b> · 7h 07m',
      gif: '<b>173</b> candidates · <b>544</b> proposals · <b>$35.75</b> · <b>85%</b> cached',
    },
    actions: [
      { type: 'unring' },
      { type: 'scrollTo', sel: '.tiles' },
    ],
    asserts: [
      /* The candidate count the caption now leads with. The hero figure is where the
         page states it, and the label is asserted with it so the number cannot be
         witnessed by some other 173: the string occurs all over this page -- the
         "Awaiting measurement" tile is 173 too -- so a document-wide substring check
         would pass with the candidate count gone. */
      { type: 'textMatches', sel: '.hero .fig', pattern: '^173$' },
      { type: 'textMatches', sel: '.hero .figlab', pattern: 'optimisation candidates' },
      { type: 'domContains', text: '$35.75' },
      { type: 'domContains', text: '266.42M' },
      { type: 'domContains', text: '7h 07m' },
      { type: 'textMatches', sel: 'div.tile:has-text("Proposals") .n', pattern: '^544$' },
    ],
  },
  {
    /* Was the cold open, back when something outside the beat loop had already parked
       the page on the board. The overview opens the demo now and this beat follows it,
       so it scrolls itself to the leaderboard, and the id says what it reveals. */
    id: 'leaderboard-reveal',
    cuts: BOTH,
    dwell: { full: 2.5, gif: 2.0 },
    caption: {
      full: '<b>173</b> ranked optimisation candidates, from one run over <code>vllm</code>',
      gif: '<b>173</b> ranked optimisation candidates',
    },
    actions: [{ type: 'scrollTo', sel: '#lbtable' }],
    asserts: [
      { type: 'visible', sel: '#lbtable' },
      { type: 'textMatches', sel: 'h2', pattern: 'Candidate leaderboard' },
      /* The caption quotes 173, so the beat proves the board still says so. */
      { type: 'textMatches', sel: '.sechead h2', pattern: '173 ranked' },
    ],
  },
  {
    id: 'run-shape',
    cuts: FULL_ONLY,
    dwell: { full: 5.0 },
    caption: { full: '<b>8 modules</b> walked — and every chart has a table view' },
    actions: [
      { type: 'scrollTo', sel: 'button.tvbtn[data-target="tv-mods"]' },
      { type: 'click', sel: 'button.tvbtn[data-target="tv-mods"]' },
      { type: 'pause', ms: 800 },
    ],
    asserts: [
      { type: 'attr', sel: 'button.tvbtn[data-target="tv-mods"]', name: 'aria-expanded', equals: 'true' },
      { type: 'visible', sel: '#tv-mods' },
    ],
  },
  {
    id: 'agents',
    cuts: FULL_ONLY,
    dwell: { full: 5.0 },
    caption: { full: 'two coding agents · <b>221 of 544</b> proposals (<b>41%</b>) cite a finding' },
    actions: [{ type: 'scrollTo', sel: 'button.tvbtn[data-target="tv-rstage"]' }],
    asserts: [
      { type: 'textMatches', sel: 'div.tile:has-text("Grounded proposals") .n', pattern: '^221$' },
      { type: 'domContains', text: 'of proposals cite a finding' },
      { type: 'domContains', text: 'Which agent did what' },
    ],
  },
  {
    id: 'source-payoff',
    cuts: FULL_ONLY,
    dwell: { full: 6.0 },
    caption: {
      full: 'what it read, and which sources <b>paid off</b>: 27 paper findings, <b>21</b> produced a proposal',
    },
    actions: [
      { type: 'scrollTo', sel: '.bar[data-tip="paper"]' },
      { type: 'hover', sel: '.bar[data-tip="paper"]' },
      { type: 'pause', ms: 900 },
    ],
    asserts: [
      { type: 'visible', sel: '#tip' },
      { type: 'textMatches', sel: '#tip', pattern: '27 findings' },
      { type: 'textMatches', sel: '#tip', pattern: '21 produced a proposal' },
    ],
  },
  {
    id: 'pair-grid',
    cuts: FULL_ONLY,
    dwell: { full: 4.0 },
    caption: { full: '<b>2,064</b> pairs evaluated · <b>10.7%</b> yielded a proposal' },
    actions: [{ type: 'scrollTo', sel: 'button.tvbtn[data-target="tv-rgrid"]' }],
    asserts: [
      { type: 'domContains', text: '2,064' },
      { type: 'domContains', text: '10.7%' },
    ],
  },
  {
    id: 'findings',
    cuts: FULL_ONLY,
    dwell: { full: 4.0 },
    caption: { full: '<b>97</b> findings over 91 distinct works, filterable' },
    actions: [
      { type: 'scrollTo', sel: '#fq' },
      { type: 'fill', sel: '#fq', text: 'cache' },
      { type: 'pause', ms: 900 },
      { type: 'fill', sel: '#fq', text: '' },
    ],
    asserts: [
      { type: 'domContains', text: '97 findings' },
      { type: 'textMatches', sel: '#fcount', pattern: '\\d' },
    ],
  },
  {
    id: 'board',
    cuts: BOTH,
    dwell: { full: 8.0, gif: 3.0 },
    caption: {
      full: 'hot modules: <b>4 of 8</b> modules hold <b>41 of the 60</b> high-impact candidates',
      gif: 'hot modules: <b>4 of 8</b> hold <b>41 of 60</b> high-impact',
    },
    actions: [
      { type: 'scrollTo', sel: '#hotbtn' },
      { type: 'ring', sel: '#hotbtn' },
      { type: 'click', sel: '#hotbtn' },
      { type: 'pause', ms: 1200 },
      { type: 'unring' },
      { type: 'select', sel: '#mod', value: 'vllm/v1/kv_offload' },
      { type: 'pause', ms: 900 },
    ],
    asserts: [
      { type: 'attr', sel: '#hotbtn', name: 'aria-pressed', equals: 'true' },
      /* #lbnote is a transient toast, set only by the deep-link reveal and cleared
         by any filter change -- including this beat's own select -- so it can never
         witness hot mode. #count is the live proof the filter took effect, and the
         caption's own figures are asserted against the prose that states them. */
      { type: 'textMatches', sel: '#count', pattern: '^\\d+ / 173 shown$' },
      { type: 'domContains', text: '4 of 8 modules, holding 41 of the 60 high-impact candidates' },
      { type: 'visible', sel: DRILL_IN_ROW },
    ],
  },
  {
    id: 'drill-writeup',
    cuts: BOTH,
    dwell: { full: 5.0, gif: 2.5 },
    caption: {
      full: 'every candidate is a full write-up — what it is, how it works today, why it is a candidate',
      gif: 'every candidate is a full write-up',
    },
    actions: [
      { type: 'scrollTo', sel: `${DRILL_IN_ROW} tr.row` },
      { type: 'click', sel: `${DRILL_IN_ROW} tr.row` },
      { type: 'pause', ms: 900 },
    ],
    asserts: [
      { type: 'visible', sel: `${DRILL_IN_ROW} tr.detail` },
      { type: 'textMatches', sel: `${DRILL_IN_ROW} .d-grid`, pattern: 'How it works today' },
      { type: 'textMatches', sel: `${DRILL_IN_ROW} .sym`, pattern: '_initiate_promotion' },
    ],
  },
  {
    id: 'drill-proposals',
    cuts: BOTH,
    dwell: { full: 4.0, gif: 3.0 },
    caption: {
      full: '<b>11 proposals</b>, 5 grounded in papers — arXiv, USENIX, AAAI',
      gif: '<b>11 proposals</b>, 5 from papers',
    },
    actions: [
      { type: 'scrollTo', sel: `${DRILL_IN_ROW} .d-props` },
      { type: 'pause', ms: 600 },
    ],
    asserts: [
      { type: 'textMatches', sel: `${DRILL_IN_ROW} .d-props`, pattern: '11 proposals' },
      { type: 'textMatches', sel: `${DRILL_IN_ROW} .d-props`, pattern: 'from paper:' },
      { type: 'minChildren', sel: `${DRILL_IN_ROW} .d-props`, n: 11 },
    ],
  },
  {
    id: 'radial-tree',
    cuts: BOTH,
    dwell: { full: 5.0, gif: 1.5 },
    caption: {
      full: 'or see all <b>173</b> as a tree',
      gif: 'or see all <b>173</b> as a tree',
    },
    actions: [
      { type: 'click', sel: `${DRILL_IN_ROW} tr.row` },
      { type: 'select', sel: '#mod', value: '' },
      { type: 'scrollTo', sel: '#lbview' },
      /* Hot mode is still on from the board beat, and the tree obeys the filters: it
         drew 86 candidates under a hub reading "hot modules" while the caption said all
         173. Invisible while the tree hung off the bottom of the frame; the moment the
         frame is filled the hub is the middle of the shot, so the filter is cleared and
         the tree really is the whole run. */
      { type: 'click', sel: '#hotbtn' },
      { type: 'click', sel: '#lbview' },
      /* The toggle sits in the section header, so the beat's scrollTo leaves the tree
         hanging off the bottom of the frame. It is drawn ~1186px tall in a 810px
         viewport, so there is no scroll position that contains it: centring it is what
         fills the frame, and it puts the hub -- the "173 candidates" label -- in the
         middle of the shot. The tree is drawn at its final size by the time the toggle's
         click resolves, so this comes *before* the beat's pause: framed last, the
         centred tree would be on screen only for the fade-out at the end of a beat whose
         choreography has already spent the dwell. */
      { type: 'scrollCenter', sel: '#rtplot svg' },
      { type: 'pause', ms: 1400 },
    ],
    asserts: [
      { type: 'visible', sel: '#lbtree' },
      { type: 'minChildren', sel: '#rtplot svg', n: 1 },
      /* Not inViewport: that demands the whole box be inside the frame, which a tree
         taller than the viewport can never satisfy. This asks the question the beat is
         actually about -- how much of the frame the tree fills. */
      { type: 'viewportCoverage', sel: '#rtplot svg', minFraction: TREE_VIEWPORT_COVERAGE },
      /* The caption says "all 173", so the beat proves no filter is still narrowing the
         tree. #count is the live count and keeps its text while the table is hidden. */
      { type: 'textMatches', sel: '#count', pattern: '^173 / 173 shown$' },
    ],
  },
  {
    id: 'close',
    cuts: BOTH,
    dwell: { full: 5.0, gif: 2.0 },
    caption: {
      full: '3 evolve scaffolds · 2 patches applied · <b>nothing measured yet</b> — the page says so',
      gif: '3 evolve scaffolds · 2 applied · <b>nothing measured yet</b>',
    },
    actions: [{ type: 'scrollTo', sel: DOWNSTREAM_SECTION }],
    asserts: [
      /* The two domContains guard the caption's figures wherever they sit in the
         document; only inViewport can witness that the closing frame is actually
         looking at the tiles, which is what the caption claims. */
      { type: 'inViewport', sel: DOWNSTREAM_SECTION },
      { type: 'domContains', text: 'Awaiting measurement' },
      { type: 'domContains', text: 'no candidate has a measured speedup yet' },
    ],
  },
];

export function beatsForCut(cut) {
  if (!CUTS.has(cut)) throw new Error(`unknown cut: ${cut}`);
  return BEATS.filter((b) => b.cuts.includes(cut));
}

export function dwellFor(beat, cut) {
  return beat.dwell[cut] ?? beat.dwell.full;
}

export function captionFor(beat, cut) {
  return beat.caption[cut] ?? beat.caption.full ?? null;
}

export function totalDuration(cut) {
  const total = beatsForCut(cut).reduce((sum, b) => sum + dwellFor(b, cut), 0);
  return Math.round(total * 10) / 10;
}

/** Returns a list of problems. Empty means the storyboard is internally consistent. */
export function validateStoryboard() {
  const problems = [];
  const seen = new Set();

  for (const beat of BEATS) {
    if (seen.has(beat.id)) problems.push(`duplicate beat id: ${beat.id}`);
    seen.add(beat.id);

    if (!beat.caption.full) problems.push(`${beat.id}: missing full caption`);
    if (typeof beat.dwell.full !== 'number') problems.push(`${beat.id}: missing full dwell`);
    if (beat.asserts.length === 0) problems.push(`${beat.id}: no assertions`);

    for (const a of beat.asserts) {
      if (a.type === 'textMatches') {
        try {
          new RegExp(a.pattern);
        } catch {
          problems.push(`${beat.id}: uncompilable pattern ${a.pattern}`);
        }
      }
    }
    for (const a of beat.actions) {
      if (a.sel === '#theme') problems.push(`${beat.id}: clicks the theme toggle`);
    }
    if (beat.cuts.includes('gif')) {
      if (!beat.caption.gif) problems.push(`${beat.id}: in gif cut but has no gif caption`);
      if (typeof beat.dwell.gif !== 'number') problems.push(`${beat.id}: in gif cut but has no gif dwell`);
    }
  }

  if (totalDuration('full') !== 62.0) problems.push(`full cut is ${totalDuration('full')}s, want 62.0s`);
  if (totalDuration('gif') !== 16.5) problems.push(`gif cut is ${totalDuration('gif')}s, want 16.5s`);

  return problems;
}
